
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


cpp_fused_convolution_backward_div_mul_native_batch_norm_backward_sum_view_0 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1280L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (1280L*x2) + (62720L*x0)), static_cast<long>(1280L), tmp3, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x1 + x1_inner + (1280L*x0))];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(tmp3 + static_cast<long>(8L*x1_inner));
                            auto tmp7 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp11 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp1 = static_cast<float>(49.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = at::vec::Vectorized<float>(tmp2);
                            auto tmp6 = tmp5 * tmp4;
                            auto tmp8 = static_cast<float>(1e-05);
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                            auto tmp10 = 1 / std::sqrt(tmp9);
                            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            tmp14.store(out_ptr3 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (62720L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1280L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1280L*x2) + (62720L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp1 = static_cast<float>(49.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 + tmp8;
                        auto tmp10 = tmp9.rsqrt();
                        auto tmp12 = tmp10 * tmp11;
                        auto tmp13 = tmp5 * tmp12;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp13.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (62720L*x0))] = tmpbuf[x1_inner]; }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4608L); x0+=static_cast<long>(8L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
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
                        auto tmp19 = static_cast<float>(1e-05);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = tmp21.rsqrt();
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        tmp25.store(in_out_ptr1 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(in_out_ptr1 + static_cast<long>(x1 + (1152L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4608L); x0+=static_cast<long>(8L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
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
                        auto tmp19 = static_cast<float>(1e-05);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = tmp21.rsqrt();
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        tmp25.store(out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_10 = async_compile.cpp('''
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (192L*x1)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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
    }
}
''')


cpp_fused_mul_native_batch_norm_backward_13 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
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
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4608L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_17 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
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
                        auto tmp19 = static_cast<float>(1e-05);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = tmp21.rsqrt();
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        tmp25.store(in_out_ptr1 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(in_out_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_19 = async_compile.cpp('''
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4608L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1152L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1152L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
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
                        auto tmp19 = static_cast<float>(1e-05);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = tmp21.rsqrt();
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        tmp25.store(out_ptr0 + static_cast<long>(x2 + (1152L*x1) + (56448L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_23 = async_compile.cpp('''
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_24 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (192L*x1)));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp8 = tmp4 * tmp7;
                    auto tmp10 = tmp4 + tmp9;
                    auto tmp13 = tmp11 - tmp12;
                    auto tmp14 = tmp10 * tmp13;
                    tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    tmp_acc2_vec = tmp_acc2_vec + tmp10;
                    tmp_acc3_vec = tmp_acc3_vec + tmp14;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
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
''')


cpp_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1152L); x0+=static_cast<long>(8L))
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
    }
}
''')


cpp_fused_mul_native_batch_norm_backward_26 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_27 = async_compile.cpp('''
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
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_29 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_30 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
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
                        auto tmp19 = static_cast<float>(1e-05);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = tmp21.rsqrt();
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        tmp25.store(in_out_ptr1 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(in_out_ptr1 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_32 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (112L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(out_ptr2 + static_cast<long>(x1 + (112L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_34 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_35 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
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
                        auto tmp19 = static_cast<float>(1e-05);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = tmp21.rsqrt();
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        tmp25.store(in_out_ptr1 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(in_out_ptr1 + static_cast<long>(x1 + (672L*x0)));
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (112L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (112L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = static_cast<float>(1e-05);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.rsqrt();
                auto tmp9 = tmp7 * tmp8;
                auto tmp10 = tmp2 * tmp9;
                tmp10.store(out_ptr0 + static_cast<long>(x1 + (112L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_39 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
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
                        auto tmp19 = static_cast<float>(1e-05);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = tmp21.rsqrt();
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        tmp25.store(out_ptr0 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_41 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_42 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (112L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (112L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (112L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (112L*x1)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (112L*x1)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp8 = tmp2 + tmp7;
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp12 = tmp8 * tmp11;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    tmp_acc2_vec = tmp_acc2_vec + tmp8;
                    tmp_acc3_vec = tmp_acc3_vec + tmp12;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
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


cpp_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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
    }
}
''')


cpp_fused_mul_native_batch_norm_backward_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (112L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (112L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (112L*x0)));
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
                tmp12.store(in_out_ptr1 + static_cast<long>(x1 + (112L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_47 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_48 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
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
                        auto tmp19 = static_cast<float>(1e-05);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = tmp21.rsqrt();
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        tmp25.store(in_out_ptr1 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(in_out_ptr1 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_50 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
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
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(1e-05);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(out_ptr2 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_52 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_53 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
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
                        auto tmp19 = static_cast<float>(1e-05);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = tmp21.rsqrt();
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        tmp25.store(in_out_ptr1 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(in_out_ptr1 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_55 = async_compile.cpp('''
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_57 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
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
                        auto tmp19 = static_cast<float>(1e-05);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = tmp21.rsqrt();
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        tmp25.store(out_ptr0 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_59 = async_compile.cpp('''
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_60 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp8 = tmp2 + tmp7;
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp12 = tmp8 * tmp11;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    tmp_acc2_vec = tmp_acc2_vec + tmp8;
                    tmp_acc3_vec = tmp_acc3_vec + tmp12;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
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


cpp_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
    }
}
''')


cpp_fused_mul_native_batch_norm_backward_62 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_63 = async_compile.cpp('''
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_66 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (240L*x1) + (47040L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (240L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (240L*x1) + (47040L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
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
                        auto tmp19 = static_cast<float>(1e-05);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = tmp21.rsqrt();
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        tmp25.store(in_out_ptr1 + static_cast<long>(x2 + (240L*x1) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(in_out_ptr1 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_68 = async_compile.cpp('''
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_69 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (240L*x1) + (188160L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (240L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (240L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (240L*x1) + (188160L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
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
                        auto tmp19 = static_cast<float>(1e-05);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = tmp21.rsqrt();
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        tmp25.store(out_ptr0 + static_cast<long>(x2 + (240L*x1) + (188160L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_72 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_73 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (40L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
    }
}
''')


cpp_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
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
    }
}
''')


cpp_fused_mul_native_batch_norm_backward_75 = async_compile.cpp('''
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
                       const float* in_ptr2)
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
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(in_out_ptr1 + static_cast<long>(x1 + (40L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (144L*x1) + (112896L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (144L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (144L*x1) + (112896L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
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
                        auto tmp19 = static_cast<float>(1e-05);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = tmp21.rsqrt();
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        tmp25.store(in_out_ptr1 + static_cast<long>(x2 + (144L*x1) + (112896L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(in_out_ptr1 + static_cast<long>(x1 + (144L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
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
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (144L*x1) + (451584L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (144L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (144L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (144L*x1) + (451584L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
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
                        auto tmp19 = static_cast<float>(1e-05);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = tmp21.rsqrt();
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        tmp25.store(out_ptr0 + static_cast<long>(x2 + (144L*x1) + (451584L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_85 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_86 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (24L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
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
    }
}
''')


cpp_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
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
    }
}
''')


cpp_fused_mul_native_batch_norm_backward_88 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
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
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(in_out_ptr1 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_90 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_92 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
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
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(96L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (96L*x1) + (301056L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (96L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (96L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (96L*x1) + (301056L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
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
                        auto tmp19 = static_cast<float>(1e-05);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = tmp21.rsqrt();
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        tmp25.store(in_out_ptr1 + static_cast<long>(x2 + (96L*x1) + (301056L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50176L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
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
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_94 = async_compile.cpp('''
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_95 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_96 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_97 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
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
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (32L*x1) + (401408L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (32L*x1) + (401408L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
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
                        auto tmp19 = static_cast<float>(1e-05);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = tmp21.rsqrt();
                        auto tmp24 = tmp22 * tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        tmp25.store(in_out_ptr1 + static_cast<long>(x2 + (32L*x1) + (401408L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(in_out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_100, primals_101, primals_103, primals_105, primals_106, primals_107, primals_108, primals_110, primals_112, primals_113, primals_114, primals_115, primals_117, primals_119, primals_120, primals_121, primals_122, primals_124, primals_126, primals_127, primals_128, primals_129, primals_131, primals_133, primals_134, primals_135, primals_136, primals_138, primals_140, primals_141, primals_142, primals_143, primals_145, primals_147, primals_148, primals_149, primals_150, primals_152, primals_154, primals_155, primals_156, primals_157, primals_159, primals_161, primals_162, primals_163, primals_164, primals_166, primals_168, primals_169, primals_170, primals_171, primals_173, primals_175, primals_176, primals_177, primals_178, primals_180, primals_182, primals_183, primals_184, primals_185, primals_187, primals_189, primals_190, primals_191, primals_192, primals_194, primals_196, primals_197, primals_198, primals_199, primals_201, primals_203, primals_204, primals_205, primals_206, primals_208, primals_210, primals_211, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, convolution, mul_3, convolution_1, add_3, mean, convolution_2, mul_8, convolution_3, mul_9, convolution_4, add_5, convolution_5, mul_16, convolution_6, add_9, mean_1, convolution_7, mul_21, convolution_8, mul_22, convolution_9, add_11, convolution_10, mul_29, convolution_11, add_15, mean_2, convolution_12, mul_34, convolution_13, mul_35, convolution_14, add_18, convolution_15, mul_42, convolution_16, add_22, mean_3, convolution_17, mul_47, convolution_18, mul_48, convolution_19, add_24, convolution_20, mul_55, convolution_21, add_28, mean_4, convolution_22, mul_60, convolution_23, mul_61, convolution_24, add_31, convolution_25, mul_68, convolution_26, add_35, mean_5, convolution_27, mul_73, convolution_28, mul_74, convolution_29, add_37, convolution_30, mul_81, convolution_31, add_41, mean_6, convolution_32, mul_86, convolution_33, mul_87, convolution_34, add_44, convolution_35, mul_94, convolution_36, add_48, mean_7, convolution_37, mul_99, convolution_38, mul_100, convolution_39, add_51, convolution_40, mul_107, convolution_41, add_55, mean_8, convolution_42, mul_112, convolution_43, mul_113, convolution_44, add_57, convolution_45, mul_120, convolution_46, add_61, mean_9, convolution_47, mul_125, convolution_48, mul_126, convolution_49, add_64, convolution_50, mul_133, convolution_51, add_68, mean_10, convolution_52, mul_138, convolution_53, mul_139, convolution_54, add_71, convolution_55, mul_146, convolution_56, add_75, mean_11, convolution_57, mul_151, convolution_58, mul_152, convolution_59, add_77, convolution_60, mul_159, convolution_61, add_81, mean_12, convolution_62, mul_164, convolution_63, mul_165, convolution_64, add_84, convolution_65, mul_172, convolution_66, add_88, mean_13, convolution_67, mul_177, convolution_68, mul_178, convolution_69, add_91, convolution_70, mul_185, convolution_71, add_95, mean_14, convolution_72, mul_190, convolution_73, mul_191, convolution_74, add_98, convolution_75, mul_198, convolution_76, add_102, mean_15, convolution_77, mul_203, convolution_78, mul_204, convolution_79, add_104, convolution_80, view, permute_1, mul_213, mul_250, mul_287, mul_324, mul_361, mul_398, mul_435, mul_472, mul_509, mul_546, mul_583, mul_620, mul_657, mul_694, mul_731, mul_768, mul_805, tangents_1 = args
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
    assert_size_stride(primals_53, (112, ), (1, ))
    assert_size_stride(primals_55, (672, ), (1, ))
    assert_size_stride(primals_57, (672, ), (1, ))
    assert_size_stride(primals_59, (112, ), (1, ))
    assert_size_stride(primals_61, (672, ), (1, ))
    assert_size_stride(primals_63, (672, ), (1, ))
    assert_size_stride(primals_65, (112, ), (1, ))
    assert_size_stride(primals_67, (672, ), (1, ))
    assert_size_stride(primals_69, (672, ), (1, ))
    assert_size_stride(primals_71, (192, ), (1, ))
    assert_size_stride(primals_73, (1152, ), (1, ))
    assert_size_stride(primals_75, (1152, ), (1, ))
    assert_size_stride(primals_77, (192, ), (1, ))
    assert_size_stride(primals_79, (1152, ), (1, ))
    assert_size_stride(primals_81, (1152, ), (1, ))
    assert_size_stride(primals_83, (192, ), (1, ))
    assert_size_stride(primals_85, (1152, ), (1, ))
    assert_size_stride(primals_87, (1152, ), (1, ))
    assert_size_stride(primals_89, (192, ), (1, ))
    assert_size_stride(primals_91, (1152, ), (1, ))
    assert_size_stride(primals_93, (1152, ), (1, ))
    assert_size_stride(primals_95, (320, ), (1, ))
    assert_size_stride(primals_97, (1280, ), (1, ))
    assert_size_stride(primals_99, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_100, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_101, (8, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_103, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_105, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_106, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_107, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_108, (4, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_110, (96, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_112, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_113, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_114, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_115, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_117, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_119, (24, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_120, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_121, (144, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_122, (6, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_124, (144, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_126, (40, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_127, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_128, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_129, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_131, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(primals_133, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_134, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_135, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_136, (10, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_138, (240, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(primals_140, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_141, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_142, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_143, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_145, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_147, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_148, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_149, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_150, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_152, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_154, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_155, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_156, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_157, (20, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_159, (480, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_161, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_162, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_163, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_164, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_166, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_168, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_169, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_170, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_171, (28, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_173, (672, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_175, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_176, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_177, (672, 1, 5, 5), (25, 25, 5, 1))
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
    assert_size_stride(primals_214, (32, ), (1, ))
    assert_size_stride(primals_215, (32, ), (1, ))
    assert_size_stride(primals_216, (32, ), (1, ))
    assert_size_stride(primals_217, (32, ), (1, ))
    assert_size_stride(primals_218, (16, ), (1, ))
    assert_size_stride(primals_219, (16, ), (1, ))
    assert_size_stride(primals_220, (96, ), (1, ))
    assert_size_stride(primals_221, (96, ), (1, ))
    assert_size_stride(primals_222, (96, ), (1, ))
    assert_size_stride(primals_223, (96, ), (1, ))
    assert_size_stride(primals_224, (24, ), (1, ))
    assert_size_stride(primals_225, (24, ), (1, ))
    assert_size_stride(primals_226, (144, ), (1, ))
    assert_size_stride(primals_227, (144, ), (1, ))
    assert_size_stride(primals_228, (144, ), (1, ))
    assert_size_stride(primals_229, (144, ), (1, ))
    assert_size_stride(primals_230, (24, ), (1, ))
    assert_size_stride(primals_231, (24, ), (1, ))
    assert_size_stride(primals_232, (144, ), (1, ))
    assert_size_stride(primals_233, (144, ), (1, ))
    assert_size_stride(primals_234, (144, ), (1, ))
    assert_size_stride(primals_235, (144, ), (1, ))
    assert_size_stride(primals_236, (40, ), (1, ))
    assert_size_stride(primals_237, (40, ), (1, ))
    assert_size_stride(primals_238, (240, ), (1, ))
    assert_size_stride(primals_239, (240, ), (1, ))
    assert_size_stride(primals_240, (240, ), (1, ))
    assert_size_stride(primals_241, (240, ), (1, ))
    assert_size_stride(primals_242, (40, ), (1, ))
    assert_size_stride(primals_243, (40, ), (1, ))
    assert_size_stride(primals_244, (240, ), (1, ))
    assert_size_stride(primals_245, (240, ), (1, ))
    assert_size_stride(primals_246, (240, ), (1, ))
    assert_size_stride(primals_247, (240, ), (1, ))
    assert_size_stride(primals_248, (80, ), (1, ))
    assert_size_stride(primals_249, (80, ), (1, ))
    assert_size_stride(primals_250, (480, ), (1, ))
    assert_size_stride(primals_251, (480, ), (1, ))
    assert_size_stride(primals_252, (480, ), (1, ))
    assert_size_stride(primals_253, (480, ), (1, ))
    assert_size_stride(primals_254, (80, ), (1, ))
    assert_size_stride(primals_255, (80, ), (1, ))
    assert_size_stride(primals_256, (480, ), (1, ))
    assert_size_stride(primals_257, (480, ), (1, ))
    assert_size_stride(primals_258, (480, ), (1, ))
    assert_size_stride(primals_259, (480, ), (1, ))
    assert_size_stride(primals_260, (80, ), (1, ))
    assert_size_stride(primals_261, (80, ), (1, ))
    assert_size_stride(primals_262, (480, ), (1, ))
    assert_size_stride(primals_263, (480, ), (1, ))
    assert_size_stride(primals_264, (480, ), (1, ))
    assert_size_stride(primals_265, (480, ), (1, ))
    assert_size_stride(primals_266, (112, ), (1, ))
    assert_size_stride(primals_267, (112, ), (1, ))
    assert_size_stride(primals_268, (672, ), (1, ))
    assert_size_stride(primals_269, (672, ), (1, ))
    assert_size_stride(primals_270, (672, ), (1, ))
    assert_size_stride(primals_271, (672, ), (1, ))
    assert_size_stride(primals_272, (112, ), (1, ))
    assert_size_stride(primals_273, (112, ), (1, ))
    assert_size_stride(primals_274, (672, ), (1, ))
    assert_size_stride(primals_275, (672, ), (1, ))
    assert_size_stride(primals_276, (672, ), (1, ))
    assert_size_stride(primals_277, (672, ), (1, ))
    assert_size_stride(primals_278, (112, ), (1, ))
    assert_size_stride(primals_279, (112, ), (1, ))
    assert_size_stride(primals_280, (672, ), (1, ))
    assert_size_stride(primals_281, (672, ), (1, ))
    assert_size_stride(primals_282, (672, ), (1, ))
    assert_size_stride(primals_283, (672, ), (1, ))
    assert_size_stride(primals_284, (192, ), (1, ))
    assert_size_stride(primals_285, (192, ), (1, ))
    assert_size_stride(primals_286, (1152, ), (1, ))
    assert_size_stride(primals_287, (1152, ), (1, ))
    assert_size_stride(primals_288, (1152, ), (1, ))
    assert_size_stride(primals_289, (1152, ), (1, ))
    assert_size_stride(primals_290, (192, ), (1, ))
    assert_size_stride(primals_291, (192, ), (1, ))
    assert_size_stride(primals_292, (1152, ), (1, ))
    assert_size_stride(primals_293, (1152, ), (1, ))
    assert_size_stride(primals_294, (1152, ), (1, ))
    assert_size_stride(primals_295, (1152, ), (1, ))
    assert_size_stride(primals_296, (192, ), (1, ))
    assert_size_stride(primals_297, (192, ), (1, ))
    assert_size_stride(primals_298, (1152, ), (1, ))
    assert_size_stride(primals_299, (1152, ), (1, ))
    assert_size_stride(primals_300, (1152, ), (1, ))
    assert_size_stride(primals_301, (1152, ), (1, ))
    assert_size_stride(primals_302, (192, ), (1, ))
    assert_size_stride(primals_303, (192, ), (1, ))
    assert_size_stride(primals_304, (1152, ), (1, ))
    assert_size_stride(primals_305, (1152, ), (1, ))
    assert_size_stride(primals_306, (1152, ), (1, ))
    assert_size_stride(primals_307, (1152, ), (1, ))
    assert_size_stride(primals_308, (320, ), (1, ))
    assert_size_stride(primals_309, (320, ), (1, ))
    assert_size_stride(primals_310, (1280, ), (1, ))
    assert_size_stride(primals_311, (1280, ), (1, ))
    assert_size_stride(primals_312, (4, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(mul_3, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_1, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(add_3, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(mean, (4, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(convolution_2, (4, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(mul_8, (4, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(convolution_3, (4, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(mul_9, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_4, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(add_5, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_5, (4, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(mul_16, (4, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(convolution_6, (4, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(add_9, (4, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(mean_1, (4, 96, 1, 1), (96, 1, 96, 96))
    assert_size_stride(convolution_7, (4, 4, 1, 1), (4, 1, 4, 4))
    assert_size_stride(mul_21, (4, 4, 1, 1), (4, 1, 4, 4))
    assert_size_stride(convolution_8, (4, 96, 1, 1), (96, 1, 96, 96))
    assert_size_stride(mul_22, (4, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(convolution_9, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(add_11, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_10, (4, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(mul_29, (4, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(convolution_11, (4, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(add_15, (4, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(mean_2, (4, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(convolution_12, (4, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(mul_34, (4, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(convolution_13, (4, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(mul_35, (4, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(convolution_14, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(add_18, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_15, (4, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(mul_42, (4, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(convolution_16, (4, 144, 28, 28), (112896, 1, 4032, 144))
    assert_size_stride(add_22, (4, 144, 28, 28), (112896, 1, 4032, 144))
    assert_size_stride(mean_3, (4, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(convolution_17, (4, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(mul_47, (4, 6, 1, 1), (6, 1, 6, 6))
    assert_size_stride(convolution_18, (4, 144, 1, 1), (144, 1, 144, 144))
    assert_size_stride(mul_48, (4, 144, 28, 28), (112896, 1, 4032, 144))
    assert_size_stride(convolution_19, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(add_24, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_20, (4, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(mul_55, (4, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(convolution_21, (4, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(add_28, (4, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(mean_4, (4, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(convolution_22, (4, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(mul_60, (4, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(convolution_23, (4, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(mul_61, (4, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(convolution_24, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(add_31, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_25, (4, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(mul_68, (4, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(convolution_26, (4, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(add_35, (4, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(mean_5, (4, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(convolution_27, (4, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(mul_73, (4, 10, 1, 1), (10, 1, 10, 10))
    assert_size_stride(convolution_28, (4, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(mul_74, (4, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(convolution_29, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(add_37, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_30, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mul_81, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_31, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(add_41, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_6, (4, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_32, (4, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(mul_86, (4, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(convolution_33, (4, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_87, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_34, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(add_44, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_35, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mul_94, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_36, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(add_48, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_7, (4, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_37, (4, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(mul_99, (4, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(convolution_38, (4, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_100, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_39, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(add_51, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_40, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mul_107, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_41, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(add_55, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_8, (4, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_42, (4, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(mul_112, (4, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(convolution_43, (4, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_113, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_44, (4, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(add_57, (4, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(convolution_45, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(mul_120, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_46, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(add_61, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(mean_9, (4, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(convolution_47, (4, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_125, (4, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_48, (4, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_126, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_49, (4, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(add_64, (4, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(convolution_50, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(mul_133, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_51, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(add_68, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(mean_10, (4, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(convolution_52, (4, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_138, (4, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_53, (4, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_139, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_54, (4, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(add_71, (4, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(convolution_55, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(mul_146, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_56, (4, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(add_75, (4, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(mean_11, (4, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(convolution_57, (4, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_151, (4, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_58, (4, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_152, (4, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(convolution_59, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(add_77, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_60, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mul_159, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_61, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(add_81, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mean_12, (4, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_62, (4, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_164, (4, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_63, (4, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_165, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_64, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(add_84, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_65, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mul_172, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_66, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(add_88, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mean_13, (4, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_67, (4, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_177, (4, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_68, (4, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_178, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_69, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(add_91, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_70, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mul_185, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_71, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(add_95, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mean_14, (4, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_72, (4, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_190, (4, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_73, (4, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_191, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_74, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(add_98, (4, 192, 7, 7), (9408, 1, 1344, 192))
    assert_size_stride(convolution_75, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mul_198, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_76, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(add_102, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mean_15, (4, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(convolution_77, (4, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(mul_203, (4, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(convolution_78, (4, 1152, 1, 1), (1152, 1, 1152, 1152))
    assert_size_stride(mul_204, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(convolution_79, (4, 320, 7, 7), (15680, 1, 2240, 320))
    assert_size_stride(add_104, (4, 320, 7, 7), (15680, 1, 2240, 320))
    assert_size_stride(convolution_80, (4, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(view, (4, 1280), (1280, 1))
    assert_size_stride(permute_1, (1000, 1280), (1280, 1))
    assert_size_stride(mul_213, (4, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(mul_250, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mul_287, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mul_324, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mul_361, (4, 1152, 7, 7), (56448, 1, 8064, 1152))
    assert_size_stride(mul_398, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(mul_435, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(mul_472, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(mul_509, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mul_546, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mul_583, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mul_620, (4, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(mul_657, (4, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(mul_694, (4, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(mul_731, (4, 144, 56, 56), (451584, 1, 8064, 144))
    assert_size_stride(mul_768, (4, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(mul_805, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    buf0 = empty((4, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf0)
    del permute_1
    buf1 = empty((1000, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), view, out=buf1)
    del view
    buf2 = empty((1000, ), device='cpu', dtype=torch.float32)
    buf3 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf4 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf5 = buf4; del buf4  # reuse
    buf6 = empty((4, 1280, 7, 7), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_mul_native_batch_norm_backward_sum_view_0(c_void_p(buf5.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(mul_213.data_ptr()), c_void_p(convolution_80.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf6.data_ptr()))
    del buf0
    del convolution_80
    del mul_213
    del primals_310
    del primals_311
    del primals_97
    del tangents_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
    buf7 = aten.convolution_backward(buf6, add_104, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_104
    del buf6
    del primals_211
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf10 = empty((320, ), device='cpu', dtype=torch.float32)
    buf11 = empty((320, ), device='cpu', dtype=torch.float32)
    buf12 = buf11; del buf11  # reuse
    buf13 = buf8; del buf8  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_1(c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(convolution_79.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf10.data_ptr()))
    del convolution_79
    del primals_308
    del primals_309
    del primals_95
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf14 = aten.convolution_backward(buf13, mul_204, primals_210, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_204
    del primals_210
    buf15 = buf14[0]
    buf16 = buf14[1]
    del buf14
    buf17 = empty_strided((4, 1152, 1, 1), (1152, 1, 4608, 4608), device='cpu', dtype=torch.float32)
    buf18 = reinterpret_tensor(buf17, (4, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf17  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_2(c_void_p(buf18.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(add_102.data_ptr()), c_void_p(convolution_78.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf19 = aten.convolution_backward(buf18, mul_203, primals_208, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf18
    del mul_203
    del primals_208
    buf20 = buf19[0]
    buf21 = buf19[1]
    buf22 = buf19[2]
    del buf19
    buf23 = reinterpret_tensor(buf20, (4, 48, 1, 1), (48, 1, 1, 1), 0); del buf20  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_3(c_void_p(buf23.data_ptr()), c_void_p(convolution_77.data_ptr()))
    del convolution_77
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf24 = aten.convolution_backward(buf23, mean_15, primals_206, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_15
    del primals_206
    buf25 = buf24[0]
    buf26 = buf24[1]
    buf27 = buf24[2]
    del buf24
    buf28 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf29 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf30 = buf29; del buf29  # reuse
    buf31 = buf15; del buf15  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_4(c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(convolution_78.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(add_102.data_ptr()), c_void_p(convolution_76.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf28.data_ptr()))
    del add_102
    del convolution_76
    del convolution_78
    del primals_306
    del primals_307
    del primals_93
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___se_gate], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
    buf32 = aten.convolution_backward(buf31, mul_198, primals_205, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1152, [True, True, False])
    del buf31
    del mul_198
    del primals_205
    buf33 = buf32[0]
    buf34 = buf32[1]
    del buf32
    buf35 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf36 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf37 = buf36; del buf36  # reuse
    buf38 = buf33; del buf33  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_5(c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(mul_250.data_ptr()), c_void_p(convolution_75.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf35.data_ptr()))
    del convolution_75
    del mul_250
    del primals_304
    del primals_305
    del primals_91
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf39 = aten.convolution_backward(buf38, add_98, primals_204, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_98
    del primals_204
    buf40 = buf39[0]
    buf41 = buf39[1]
    del buf39
    buf45 = empty_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_6(c_void_p(buf40.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf45.data_ptr()))
    del primals_89
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf46 = aten.convolution_backward(buf45, mul_191, primals_203, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_191
    del primals_203
    buf47 = buf46[0]
    buf49 = reinterpret_tensor(buf25, (4, 1152, 1, 1), (1152, 1, 4608, 4608), 0); del buf25  # reuse
    buf50 = reinterpret_tensor(buf49, (4, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf49  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_7(c_void_p(buf50.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(add_95.data_ptr()), c_void_p(convolution_73.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf51 = aten.convolution_backward(buf50, mul_190, primals_201, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf50
    del mul_190
    del primals_201
    buf52 = buf51[0]
    buf55 = reinterpret_tensor(buf52, (4, 48, 1, 1), (48, 1, 1, 1), 0); del buf52  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_8(c_void_p(buf55.data_ptr()), c_void_p(convolution_72.data_ptr()))
    del convolution_72
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf56 = aten.convolution_backward(buf55, mean_14, primals_199, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_14
    del primals_199
    buf57 = buf56[0]
    buf63 = buf38; del buf38  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_9(c_void_p(buf47.data_ptr()), c_void_p(convolution_73.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(add_95.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf63.data_ptr()))
    del primals_87
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
    buf64 = aten.convolution_backward(buf63, mul_185, primals_198, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
    del mul_185
    del primals_198
    buf65 = buf64[0]
    buf70 = buf63; del buf63  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_10(c_void_p(buf65.data_ptr()), c_void_p(mul_287.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(buf70.data_ptr()))
    del primals_85
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf71 = aten.convolution_backward(buf70, add_91, primals_197, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_91
    del buf70
    del primals_197
    buf72 = buf71[0]
    buf42 = reinterpret_tensor(buf55, (192, ), (1, ), 0); del buf55  # reuse
    buf43 = reinterpret_tensor(buf23, (192, ), (1, ), 0); del buf23  # reuse
    buf74 = empty((192, ), device='cpu', dtype=torch.float32)
    buf75 = empty((192, ), device='cpu', dtype=torch.float32)
    buf44 = buf43; del buf43  # reuse
    cpp_fused_add_native_batch_norm_backward_11(c_void_p(buf44.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(convolution_74.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(convolution_69.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()))
    del convolution_69
    del convolution_74
    del primals_296
    del primals_302
    del primals_303
    buf48 = buf46[1]
    del buf46
    buf53 = buf51[1]
    buf54 = buf51[2]
    del buf51
    buf58 = buf56[1]
    buf59 = buf56[2]
    del buf56
    buf60 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf61 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf62 = buf61; del buf61  # reuse
    cpp_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_12(c_void_p(buf62.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(convolution_73.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(add_95.data_ptr()), c_void_p(convolution_71.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(buf60.data_ptr()))
    del add_95
    del buf47
    del convolution_71
    del convolution_73
    del primals_300
    del primals_301
    buf66 = buf64[1]
    del buf64
    buf67 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf68 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf69 = buf68; del buf68  # reuse
    cpp_fused_mul_native_batch_norm_backward_13(c_void_p(buf69.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(mul_287.data_ptr()), c_void_p(convolution_70.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(buf67.data_ptr()))
    del buf65
    del convolution_70
    del mul_287
    del primals_298
    del primals_299
    buf73 = buf71[1]
    del buf71
    buf76 = buf75; del buf75  # reuse
    buf77 = buf45; del buf45  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_14(c_void_p(buf76.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf77.data_ptr()))
    del primals_297
    del primals_83
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf78 = aten.convolution_backward(buf77, mul_178, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_178
    del primals_196
    buf79 = buf78[0]
    buf80 = buf78[1]
    del buf78
    buf81 = reinterpret_tensor(buf57, (4, 1152, 1, 1), (1152, 1, 4608, 4608), 0); del buf57  # reuse
    buf82 = reinterpret_tensor(buf81, (4, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf81  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_15(c_void_p(buf82.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(add_88.data_ptr()), c_void_p(convolution_68.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf83 = aten.convolution_backward(buf82, mul_177, primals_194, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf82
    del mul_177
    del primals_194
    buf84 = buf83[0]
    buf85 = buf83[1]
    buf86 = buf83[2]
    del buf83
    buf87 = reinterpret_tensor(buf84, (4, 48, 1, 1), (48, 1, 1, 1), 0); del buf84  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_16(c_void_p(buf87.data_ptr()), c_void_p(convolution_67.data_ptr()))
    del convolution_67
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf88 = aten.convolution_backward(buf87, mean_13, primals_192, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_13
    del primals_192
    buf89 = buf88[0]
    buf90 = buf88[1]
    buf91 = buf88[2]
    del buf88
    buf92 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf93 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf94 = buf93; del buf93  # reuse
    buf95 = buf79; del buf79  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_17(c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(convolution_68.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(add_88.data_ptr()), c_void_p(convolution_66.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(buf92.data_ptr()))
    del add_88
    del convolution_66
    del convolution_68
    del primals_294
    del primals_295
    del primals_81
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
    buf96 = aten.convolution_backward(buf95, mul_172, primals_191, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
    del buf95
    del mul_172
    del primals_191
    buf97 = buf96[0]
    buf98 = buf96[1]
    del buf96
    buf99 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf100 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf101 = buf100; del buf100  # reuse
    buf102 = buf97; del buf97  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_18(c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(mul_324.data_ptr()), c_void_p(convolution_65.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf99.data_ptr()))
    del convolution_65
    del mul_324
    del primals_292
    del primals_293
    del primals_79
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf103 = aten.convolution_backward(buf102, add_84, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_84
    del primals_190
    buf104 = buf103[0]
    buf105 = buf103[1]
    del buf103
    buf109 = buf77; del buf77  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_19(c_void_p(buf40.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf109.data_ptr()))
    del primals_77
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf110 = aten.convolution_backward(buf109, mul_165, primals_189, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf109
    del mul_165
    del primals_189
    buf111 = buf110[0]
    buf113 = reinterpret_tensor(buf89, (4, 1152, 1, 1), (1152, 1, 4608, 4608), 0); del buf89  # reuse
    buf114 = reinterpret_tensor(buf113, (4, 1152, 1, 1), (1152, 1, 1, 1), 0); del buf113  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_20(c_void_p(buf114.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(add_81.data_ptr()), c_void_p(convolution_63.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf115 = aten.convolution_backward(buf114, mul_164, primals_187, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf114
    del mul_164
    del primals_187
    buf116 = buf115[0]
    buf119 = reinterpret_tensor(buf116, (4, 48, 1, 1), (48, 1, 1, 1), 0); del buf116  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_21(c_void_p(buf119.data_ptr()), c_void_p(convolution_62.data_ptr()))
    del convolution_62
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf120 = aten.convolution_backward(buf119, mean_12, primals_185, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_12
    del primals_185
    buf121 = buf120[0]
    buf127 = buf102; del buf102  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_22(c_void_p(buf111.data_ptr()), c_void_p(convolution_63.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(add_81.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf127.data_ptr()))
    del primals_75
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
    buf128 = aten.convolution_backward(buf127, mul_159, primals_184, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False])
    del mul_159
    del primals_184
    buf129 = buf128[0]
    buf134 = buf127; del buf127  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_23(c_void_p(buf129.data_ptr()), c_void_p(mul_361.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf134.data_ptr()))
    del primals_73
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf135 = aten.convolution_backward(buf134, add_77, primals_183, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_77
    del buf134
    del primals_183
    buf136 = buf135[0]
    buf106 = reinterpret_tensor(buf119, (192, ), (1, ), 0); del buf119  # reuse
    buf107 = reinterpret_tensor(buf87, (192, ), (1, ), 0); del buf87  # reuse
    buf138 = empty((192, ), device='cpu', dtype=torch.float32)
    buf139 = empty((192, ), device='cpu', dtype=torch.float32)
    buf108 = buf107; del buf107  # reuse
    cpp_fused_add_native_batch_norm_backward_24(c_void_p(buf108.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(convolution_64.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(convolution_59.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()))
    del convolution_59
    del convolution_64
    del primals_284
    del primals_290
    del primals_291
    buf112 = buf110[1]
    del buf110
    buf117 = buf115[1]
    buf118 = buf115[2]
    del buf115
    buf122 = buf120[1]
    buf123 = buf120[2]
    del buf120
    buf124 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf125 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf126 = buf125; del buf125  # reuse
    cpp_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_25(c_void_p(buf126.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(convolution_63.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(add_81.data_ptr()), c_void_p(convolution_61.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(buf124.data_ptr()))
    del add_81
    del buf111
    del buf121
    del convolution_61
    del convolution_63
    del primals_288
    del primals_289
    buf130 = buf128[1]
    del buf128
    buf131 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf132 = empty((1152, ), device='cpu', dtype=torch.float32)
    buf133 = buf132; del buf132  # reuse
    cpp_fused_mul_native_batch_norm_backward_26(c_void_p(buf133.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(mul_361.data_ptr()), c_void_p(convolution_60.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(buf131.data_ptr()))
    del buf129
    del convolution_60
    del mul_361
    del primals_286
    del primals_287
    buf137 = buf135[1]
    del buf135
    buf140 = buf139; del buf139  # reuse
    buf141 = buf104; del buf104  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_27(c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(primals_71.data_ptr()))
    del buf136
    del buf40
    del buf72
    del primals_285
    del primals_71
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf142 = aten.convolution_backward(buf141, mul_152, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf141
    del mul_152
    del primals_182
    buf143 = buf142[0]
    buf144 = buf142[1]
    del buf142
    buf145 = empty_strided((4, 672, 1, 1), (672, 1, 2688, 2688), device='cpu', dtype=torch.float32)
    buf146 = reinterpret_tensor(buf145, (4, 672, 1, 1), (672, 1, 1, 1), 0); del buf145  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_28(c_void_p(buf146.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(add_75.data_ptr()), c_void_p(convolution_58.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf147 = aten.convolution_backward(buf146, mul_151, primals_180, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf146
    del mul_151
    del primals_180
    buf148 = buf147[0]
    buf149 = buf147[1]
    buf150 = buf147[2]
    del buf147
    buf151 = reinterpret_tensor(buf148, (4, 28, 1, 1), (28, 1, 1, 1), 0); del buf148  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_29(c_void_p(buf151.data_ptr()), c_void_p(convolution_57.data_ptr()))
    del convolution_57
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf152 = aten.convolution_backward(buf151, mean_11, primals_178, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_11
    del primals_178
    buf153 = buf152[0]
    buf154 = buf152[1]
    buf155 = buf152[2]
    del buf152
    buf156 = empty((672, ), device='cpu', dtype=torch.float32)
    buf157 = empty((672, ), device='cpu', dtype=torch.float32)
    buf158 = buf157; del buf157  # reuse
    buf159 = buf143; del buf143  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_30(c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(convolution_58.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(add_75.data_ptr()), c_void_p(convolution_56.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf156.data_ptr()))
    del add_75
    del convolution_56
    del convolution_58
    del primals_282
    del primals_283
    del primals_69
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
    buf160 = aten.convolution_backward(buf159, mul_146, primals_177, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
    del buf159
    del mul_146
    del primals_177
    buf161 = buf160[0]
    buf162 = buf160[1]
    del buf160
    buf163 = empty((672, ), device='cpu', dtype=torch.float32)
    buf164 = empty((672, ), device='cpu', dtype=torch.float32)
    buf165 = buf164; del buf164  # reuse
    buf166 = buf161; del buf161  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_31(c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(mul_398.data_ptr()), c_void_p(convolution_55.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf163.data_ptr()))
    del convolution_55
    del mul_398
    del primals_280
    del primals_281
    del primals_67
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf167 = aten.convolution_backward(buf166, add_71, primals_176, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_71
    del buf166
    del primals_176
    buf168 = buf167[0]
    buf169 = buf167[1]
    del buf167
    buf170 = reinterpret_tensor(buf151, (112, ), (1, ), 0); del buf151  # reuse
    buf171 = empty((112, ), device='cpu', dtype=torch.float32)
    buf172 = buf171; del buf171  # reuse
    buf173 = empty_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_32(c_void_p(buf172.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf173.data_ptr()))
    del convolution_54
    del primals_278
    del primals_279
    del primals_65
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf174 = aten.convolution_backward(buf173, mul_139, primals_175, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_139
    del primals_175
    buf175 = buf174[0]
    buf176 = buf174[1]
    del buf174
    buf177 = reinterpret_tensor(buf153, (4, 672, 1, 1), (672, 1, 2688, 2688), 0); del buf153  # reuse
    buf178 = reinterpret_tensor(buf177, (4, 672, 1, 1), (672, 1, 1, 1), 0); del buf177  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_33(c_void_p(buf178.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(add_68.data_ptr()), c_void_p(convolution_53.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf179 = aten.convolution_backward(buf178, mul_138, primals_173, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf178
    del mul_138
    del primals_173
    buf180 = buf179[0]
    buf181 = buf179[1]
    buf182 = buf179[2]
    del buf179
    buf183 = reinterpret_tensor(buf180, (4, 28, 1, 1), (28, 1, 1, 1), 0); del buf180  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_34(c_void_p(buf183.data_ptr()), c_void_p(convolution_52.data_ptr()))
    del convolution_52
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf184 = aten.convolution_backward(buf183, mean_10, primals_171, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_10
    del primals_171
    buf185 = buf184[0]
    buf186 = buf184[1]
    buf187 = buf184[2]
    del buf184
    buf188 = empty((672, ), device='cpu', dtype=torch.float32)
    buf189 = empty((672, ), device='cpu', dtype=torch.float32)
    buf190 = buf189; del buf189  # reuse
    buf191 = buf175; del buf175  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_35(c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(convolution_53.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(add_68.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf188.data_ptr()))
    del add_68
    del convolution_51
    del convolution_53
    del primals_276
    del primals_277
    del primals_63
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
    buf192 = aten.convolution_backward(buf191, mul_133, primals_170, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
    del buf191
    del mul_133
    del primals_170
    buf193 = buf192[0]
    buf194 = buf192[1]
    del buf192
    buf195 = empty((672, ), device='cpu', dtype=torch.float32)
    buf196 = empty((672, ), device='cpu', dtype=torch.float32)
    buf197 = buf196; del buf196  # reuse
    buf198 = buf193; del buf193  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_36(c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(mul_435.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf195.data_ptr()))
    del convolution_50
    del mul_435
    del primals_274
    del primals_275
    del primals_61
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf199 = aten.convolution_backward(buf198, add_64, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_64
    del primals_169
    buf200 = buf199[0]
    buf201 = buf199[1]
    del buf199
    buf205 = buf173; del buf173  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_37(c_void_p(buf168.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf205.data_ptr()))
    del primals_59
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf206 = aten.convolution_backward(buf205, mul_126, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf205
    del mul_126
    del primals_168
    buf207 = buf206[0]
    buf209 = reinterpret_tensor(buf185, (4, 672, 1, 1), (672, 1, 2688, 2688), 0); del buf185  # reuse
    buf210 = reinterpret_tensor(buf209, (4, 672, 1, 1), (672, 1, 1, 1), 0); del buf209  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_38(c_void_p(buf210.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(add_61.data_ptr()), c_void_p(convolution_48.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf211 = aten.convolution_backward(buf210, mul_125, primals_166, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf210
    del mul_125
    del primals_166
    buf212 = buf211[0]
    buf215 = reinterpret_tensor(buf212, (4, 28, 1, 1), (28, 1, 1, 1), 0); del buf212  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_39(c_void_p(buf215.data_ptr()), c_void_p(convolution_47.data_ptr()))
    del convolution_47
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf216 = aten.convolution_backward(buf215, mean_9, primals_164, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_9
    del primals_164
    buf217 = buf216[0]
    buf223 = buf198; del buf198  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_40(c_void_p(buf207.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(add_61.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf223.data_ptr()))
    del primals_57
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
    buf224 = aten.convolution_backward(buf223, mul_120, primals_163, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
    del mul_120
    del primals_163
    buf225 = buf224[0]
    buf230 = buf223; del buf223  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_41(c_void_p(buf225.data_ptr()), c_void_p(mul_472.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf230.data_ptr()))
    del primals_55
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf231 = aten.convolution_backward(buf230, add_57, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_57
    del buf230
    del primals_162
    buf232 = buf231[0]
    buf202 = reinterpret_tensor(buf215, (112, ), (1, ), 0); del buf215  # reuse
    buf203 = reinterpret_tensor(buf183, (112, ), (1, ), 0); del buf183  # reuse
    buf234 = empty((112, ), device='cpu', dtype=torch.float32)
    buf235 = empty((112, ), device='cpu', dtype=torch.float32)
    buf204 = buf203; del buf203  # reuse
    cpp_fused_add_native_batch_norm_backward_42(c_void_p(buf204.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()))
    del convolution_44
    del convolution_49
    del primals_266
    del primals_272
    del primals_273
    buf208 = buf206[1]
    del buf206
    buf213 = buf211[1]
    buf214 = buf211[2]
    del buf211
    buf218 = buf216[1]
    buf219 = buf216[2]
    del buf216
    buf220 = empty((672, ), device='cpu', dtype=torch.float32)
    buf221 = empty((672, ), device='cpu', dtype=torch.float32)
    buf222 = buf221; del buf221  # reuse
    cpp_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_43(c_void_p(buf222.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(add_61.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(buf220.data_ptr()))
    del add_61
    del buf207
    del buf217
    del convolution_46
    del convolution_48
    del primals_270
    del primals_271
    buf226 = buf224[1]
    del buf224
    buf227 = empty((672, ), device='cpu', dtype=torch.float32)
    buf228 = empty((672, ), device='cpu', dtype=torch.float32)
    buf229 = buf228; del buf228  # reuse
    cpp_fused_mul_native_batch_norm_backward_44(c_void_p(buf229.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(mul_472.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(buf227.data_ptr()))
    del buf225
    del convolution_45
    del mul_472
    del primals_268
    del primals_269
    buf233 = buf231[1]
    del buf231
    buf236 = buf235; del buf235  # reuse
    buf237 = buf168; del buf168  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_45(c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(primals_53.data_ptr()))
    del buf200
    del buf232
    del primals_267
    del primals_53
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf238 = aten.convolution_backward(buf237, mul_113, primals_161, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf237
    del mul_113
    del primals_161
    buf239 = buf238[0]
    buf240 = buf238[1]
    del buf238
    buf241 = empty_strided((4, 480, 1, 1), (480, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf242 = reinterpret_tensor(buf241, (4, 480, 1, 1), (480, 1, 1, 1), 0); del buf241  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_46(c_void_p(buf242.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(add_55.data_ptr()), c_void_p(convolution_43.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf243 = aten.convolution_backward(buf242, mul_112, primals_159, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf242
    del mul_112
    del primals_159
    buf244 = buf243[0]
    buf245 = buf243[1]
    buf246 = buf243[2]
    del buf243
    buf247 = reinterpret_tensor(buf244, (4, 20, 1, 1), (20, 1, 1, 1), 0); del buf244  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_47(c_void_p(buf247.data_ptr()), c_void_p(convolution_42.data_ptr()))
    del convolution_42
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf248 = aten.convolution_backward(buf247, mean_8, primals_157, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_8
    del primals_157
    buf249 = buf248[0]
    buf250 = buf248[1]
    buf251 = buf248[2]
    del buf248
    buf252 = empty((480, ), device='cpu', dtype=torch.float32)
    buf253 = empty((480, ), device='cpu', dtype=torch.float32)
    buf254 = buf253; del buf253  # reuse
    buf255 = buf239; del buf239  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_48(c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(add_55.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf252.data_ptr()))
    del add_55
    del convolution_41
    del convolution_43
    del primals_264
    del primals_265
    del primals_51
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
    buf256 = aten.convolution_backward(buf255, mul_107, primals_156, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False])
    del buf255
    del mul_107
    del primals_156
    buf257 = buf256[0]
    buf258 = buf256[1]
    del buf256
    buf259 = empty((480, ), device='cpu', dtype=torch.float32)
    buf260 = empty((480, ), device='cpu', dtype=torch.float32)
    buf261 = buf260; del buf260  # reuse
    buf262 = buf257; del buf257  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_49(c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(mul_509.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf259.data_ptr()))
    del convolution_40
    del mul_509
    del primals_262
    del primals_263
    del primals_49
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf263 = aten.convolution_backward(buf262, add_51, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_51
    del buf262
    del primals_155
    buf264 = buf263[0]
    buf265 = buf263[1]
    del buf263
    buf266 = reinterpret_tensor(buf247, (80, ), (1, ), 0); del buf247  # reuse
    buf267 = empty((80, ), device='cpu', dtype=torch.float32)
    buf268 = buf267; del buf267  # reuse
    buf269 = reinterpret_tensor(buf13, (4, 80, 14, 14), (15680, 1, 1120, 80), 0); del buf13  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_50(c_void_p(buf268.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf269.data_ptr()))
    del convolution_39
    del primals_260
    del primals_261
    del primals_47
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf270 = aten.convolution_backward(buf269, mul_100, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_100
    del primals_154
    buf271 = buf270[0]
    buf272 = buf270[1]
    del buf270
    buf273 = reinterpret_tensor(buf249, (4, 480, 1, 1), (480, 1, 1920, 1920), 0); del buf249  # reuse
    buf274 = reinterpret_tensor(buf273, (4, 480, 1, 1), (480, 1, 1, 1), 0); del buf273  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_51(c_void_p(buf274.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(add_48.data_ptr()), c_void_p(convolution_38.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf275 = aten.convolution_backward(buf274, mul_99, primals_152, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf274
    del mul_99
    del primals_152
    buf276 = buf275[0]
    buf277 = buf275[1]
    buf278 = buf275[2]
    del buf275
    buf279 = reinterpret_tensor(buf276, (4, 20, 1, 1), (20, 1, 1, 1), 0); del buf276  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_52(c_void_p(buf279.data_ptr()), c_void_p(convolution_37.data_ptr()))
    del convolution_37
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf280 = aten.convolution_backward(buf279, mean_7, primals_150, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_7
    del primals_150
    buf281 = buf280[0]
    buf282 = buf280[1]
    buf283 = buf280[2]
    del buf280
    buf284 = empty((480, ), device='cpu', dtype=torch.float32)
    buf285 = empty((480, ), device='cpu', dtype=torch.float32)
    buf286 = buf285; del buf285  # reuse
    buf287 = buf271; del buf271  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_53(c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(add_48.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf284.data_ptr()))
    del add_48
    del convolution_36
    del convolution_38
    del primals_258
    del primals_259
    del primals_45
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
    buf288 = aten.convolution_backward(buf287, mul_94, primals_149, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
    del buf287
    del mul_94
    del primals_149
    buf289 = buf288[0]
    buf290 = buf288[1]
    del buf288
    buf291 = empty((480, ), device='cpu', dtype=torch.float32)
    buf292 = empty((480, ), device='cpu', dtype=torch.float32)
    buf293 = buf292; del buf292  # reuse
    buf294 = buf289; del buf289  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_54(c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(mul_546.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf291.data_ptr()))
    del convolution_35
    del mul_546
    del primals_256
    del primals_257
    del primals_43
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf295 = aten.convolution_backward(buf294, add_44, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_44
    del primals_148
    buf296 = buf295[0]
    buf297 = buf295[1]
    del buf295
    buf301 = buf269; del buf269  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_55(c_void_p(buf264.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf301.data_ptr()))
    del primals_41
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf302 = aten.convolution_backward(buf301, mul_87, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf301
    del mul_87
    del primals_147
    buf303 = buf302[0]
    buf305 = reinterpret_tensor(buf281, (4, 480, 1, 1), (480, 1, 1920, 1920), 0); del buf281  # reuse
    buf306 = reinterpret_tensor(buf305, (4, 480, 1, 1), (480, 1, 1, 1), 0); del buf305  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_56(c_void_p(buf306.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(add_41.data_ptr()), c_void_p(convolution_33.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf307 = aten.convolution_backward(buf306, mul_86, primals_145, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf306
    del mul_86
    del primals_145
    buf308 = buf307[0]
    buf311 = reinterpret_tensor(buf308, (4, 20, 1, 1), (20, 1, 1, 1), 0); del buf308  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_57(c_void_p(buf311.data_ptr()), c_void_p(convolution_32.data_ptr()))
    del convolution_32
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf312 = aten.convolution_backward(buf311, mean_6, primals_143, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_6
    del primals_143
    buf313 = buf312[0]
    buf319 = buf294; del buf294  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_58(c_void_p(buf303.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(add_41.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf319.data_ptr()))
    del primals_39
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
    buf320 = aten.convolution_backward(buf319, mul_81, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
    del mul_81
    del primals_142
    buf321 = buf320[0]
    buf326 = buf319; del buf319  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_59(c_void_p(buf321.data_ptr()), c_void_p(mul_583.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf326.data_ptr()))
    del primals_37
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf327 = aten.convolution_backward(buf326, add_37, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_37
    del buf326
    del primals_141
    buf328 = buf327[0]
    buf298 = reinterpret_tensor(buf311, (80, ), (1, ), 0); del buf311  # reuse
    buf299 = reinterpret_tensor(buf279, (80, ), (1, ), 0); del buf279  # reuse
    buf330 = empty((80, ), device='cpu', dtype=torch.float32)
    buf331 = empty((80, ), device='cpu', dtype=torch.float32)
    buf300 = buf299; del buf299  # reuse
    cpp_fused_add_native_batch_norm_backward_60(c_void_p(buf300.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()))
    del convolution_29
    del convolution_34
    del primals_248
    del primals_254
    del primals_255
    buf304 = buf302[1]
    del buf302
    buf309 = buf307[1]
    buf310 = buf307[2]
    del buf307
    buf314 = buf312[1]
    buf315 = buf312[2]
    del buf312
    buf316 = empty((480, ), device='cpu', dtype=torch.float32)
    buf317 = empty((480, ), device='cpu', dtype=torch.float32)
    buf318 = buf317; del buf317  # reuse
    cpp_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_61(c_void_p(buf318.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(add_41.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(buf316.data_ptr()))
    del add_41
    del buf303
    del buf313
    del convolution_31
    del convolution_33
    del primals_252
    del primals_253
    buf322 = buf320[1]
    del buf320
    buf323 = empty((480, ), device='cpu', dtype=torch.float32)
    buf324 = empty((480, ), device='cpu', dtype=torch.float32)
    buf325 = buf324; del buf324  # reuse
    cpp_fused_mul_native_batch_norm_backward_62(c_void_p(buf325.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(mul_583.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(buf323.data_ptr()))
    del buf321
    del convolution_30
    del mul_583
    del primals_250
    del primals_251
    buf329 = buf327[1]
    del buf327
    buf332 = buf331; del buf331  # reuse
    buf333 = buf264; del buf264  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_63(c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(primals_35.data_ptr()))
    del buf296
    del buf328
    del primals_249
    del primals_35
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf334 = aten.convolution_backward(buf333, mul_74, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf333
    del mul_74
    del primals_140
    buf335 = buf334[0]
    buf336 = buf334[1]
    del buf334
    buf337 = empty_strided((4, 240, 1, 1), (240, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf338 = reinterpret_tensor(buf337, (4, 240, 1, 1), (240, 1, 1, 1), 0); del buf337  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_64(c_void_p(buf338.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(add_35.data_ptr()), c_void_p(convolution_28.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf339 = aten.convolution_backward(buf338, mul_73, primals_138, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf338
    del mul_73
    del primals_138
    buf340 = buf339[0]
    buf341 = buf339[1]
    buf342 = buf339[2]
    del buf339
    buf343 = reinterpret_tensor(buf340, (4, 10, 1, 1), (10, 1, 1, 1), 0); del buf340  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_65(c_void_p(buf343.data_ptr()), c_void_p(convolution_27.data_ptr()))
    del convolution_27
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf344 = aten.convolution_backward(buf343, mean_5, primals_136, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_5
    del primals_136
    buf345 = buf344[0]
    buf346 = buf344[1]
    buf347 = buf344[2]
    del buf344
    buf348 = empty((240, ), device='cpu', dtype=torch.float32)
    buf349 = empty((240, ), device='cpu', dtype=torch.float32)
    buf350 = buf349; del buf349  # reuse
    buf351 = buf335; del buf335  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_66(c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(add_35.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf348.data_ptr()))
    del add_35
    del convolution_26
    del convolution_28
    del primals_246
    del primals_247
    del primals_33
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
    buf352 = aten.convolution_backward(buf351, mul_68, primals_135, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False])
    del buf351
    del mul_68
    del primals_135
    buf353 = buf352[0]
    buf354 = buf352[1]
    del buf352
    buf355 = empty((240, ), device='cpu', dtype=torch.float32)
    buf356 = empty((240, ), device='cpu', dtype=torch.float32)
    buf357 = buf356; del buf356  # reuse
    buf358 = buf353; del buf353  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_67(c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(mul_620.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(primals_245.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf355.data_ptr()))
    del convolution_25
    del mul_620
    del primals_244
    del primals_245
    del primals_31
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf359 = aten.convolution_backward(buf358, add_31, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_31
    del primals_134
    buf360 = buf359[0]
    buf361 = buf359[1]
    del buf359
    buf365 = empty_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_68(c_void_p(buf360.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf365.data_ptr()))
    del primals_29
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf366 = aten.convolution_backward(buf365, mul_61, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf365
    del mul_61
    del primals_133
    buf367 = buf366[0]
    buf369 = reinterpret_tensor(buf345, (4, 240, 1, 1), (240, 1, 960, 960), 0); del buf345  # reuse
    buf370 = reinterpret_tensor(buf369, (4, 240, 1, 1), (240, 1, 1, 1), 0); del buf369  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_69(c_void_p(buf370.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(add_28.data_ptr()), c_void_p(convolution_23.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf371 = aten.convolution_backward(buf370, mul_60, primals_131, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf370
    del mul_60
    del primals_131
    buf372 = buf371[0]
    buf375 = reinterpret_tensor(buf372, (4, 10, 1, 1), (10, 1, 1, 1), 0); del buf372  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_70(c_void_p(buf375.data_ptr()), c_void_p(convolution_22.data_ptr()))
    del convolution_22
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf376 = aten.convolution_backward(buf375, mean_4, primals_129, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_4
    del primals_129
    buf377 = buf376[0]
    buf383 = buf358; del buf358  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_71(c_void_p(buf367.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(add_28.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf383.data_ptr()))
    del primals_27
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
    buf384 = aten.convolution_backward(buf383, mul_55, primals_128, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False])
    del mul_55
    del primals_128
    buf385 = buf384[0]
    buf390 = buf383; del buf383  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_72(c_void_p(buf385.data_ptr()), c_void_p(mul_657.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf390.data_ptr()))
    del primals_25
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf391 = aten.convolution_backward(buf390, add_24, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_24
    del buf390
    del primals_127
    buf392 = buf391[0]
    buf362 = reinterpret_tensor(buf375, (40, ), (1, ), 0); del buf375  # reuse
    buf363 = reinterpret_tensor(buf343, (40, ), (1, ), 0); del buf343  # reuse
    buf394 = empty((40, ), device='cpu', dtype=torch.float32)
    buf395 = empty((40, ), device='cpu', dtype=torch.float32)
    buf364 = buf363; del buf363  # reuse
    cpp_fused_add_native_batch_norm_backward_73(c_void_p(buf364.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()))
    del convolution_19
    del convolution_24
    del primals_236
    del primals_242
    del primals_243
    buf368 = buf366[1]
    del buf366
    buf373 = buf371[1]
    buf374 = buf371[2]
    del buf371
    buf378 = buf376[1]
    buf379 = buf376[2]
    del buf376
    buf380 = empty((240, ), device='cpu', dtype=torch.float32)
    buf381 = empty((240, ), device='cpu', dtype=torch.float32)
    buf382 = buf381; del buf381  # reuse
    cpp_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_74(c_void_p(buf382.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(add_28.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(buf380.data_ptr()))
    del add_28
    del buf367
    del buf377
    del convolution_21
    del convolution_23
    del primals_240
    del primals_241
    buf386 = buf384[1]
    del buf384
    buf387 = empty((240, ), device='cpu', dtype=torch.float32)
    buf388 = empty((240, ), device='cpu', dtype=torch.float32)
    buf389 = buf388; del buf388  # reuse
    cpp_fused_mul_native_batch_norm_backward_75(c_void_p(buf389.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(mul_657.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(buf387.data_ptr()))
    del buf385
    del convolution_20
    del mul_657
    del primals_238
    del primals_239
    buf393 = buf391[1]
    del buf391
    buf396 = buf395; del buf395  # reuse
    buf397 = buf360; del buf360  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_76(c_void_p(buf396.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(primals_23.data_ptr()))
    del buf392
    del primals_23
    del primals_237
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf398 = aten.convolution_backward(buf397, mul_48, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf397
    del mul_48
    del primals_126
    buf399 = buf398[0]
    buf400 = buf398[1]
    del buf398
    buf401 = empty_strided((4, 144, 1, 1), (144, 1, 576, 576), device='cpu', dtype=torch.float32)
    buf402 = reinterpret_tensor(buf401, (4, 144, 1, 1), (144, 1, 1, 1), 0); del buf401  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_77(c_void_p(buf402.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(add_22.data_ptr()), c_void_p(convolution_18.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf403 = aten.convolution_backward(buf402, mul_47, primals_124, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf402
    del mul_47
    del primals_124
    buf404 = buf403[0]
    buf405 = buf403[1]
    buf406 = buf403[2]
    del buf403
    buf407 = reinterpret_tensor(buf404, (4, 6, 1, 1), (6, 1, 1, 1), 0); del buf404  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_78(c_void_p(buf407.data_ptr()), c_void_p(convolution_17.data_ptr()))
    del convolution_17
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf408 = aten.convolution_backward(buf407, mean_3, primals_122, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_3
    del primals_122
    buf409 = buf408[0]
    buf410 = buf408[1]
    buf411 = buf408[2]
    del buf408
    buf412 = empty((144, ), device='cpu', dtype=torch.float32)
    buf413 = empty((144, ), device='cpu', dtype=torch.float32)
    buf414 = buf413; del buf413  # reuse
    buf415 = buf399; del buf399  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_79(c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(add_22.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf412.data_ptr()))
    del add_22
    del convolution_16
    del convolution_18
    del primals_21
    del primals_234
    del primals_235
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
    buf416 = aten.convolution_backward(buf415, mul_42, primals_121, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 144, [True, True, False])
    del buf415
    del mul_42
    del primals_121
    buf417 = buf416[0]
    buf418 = buf416[1]
    del buf416
    buf419 = empty((144, ), device='cpu', dtype=torch.float32)
    buf420 = empty((144, ), device='cpu', dtype=torch.float32)
    buf421 = buf420; del buf420  # reuse
    buf422 = buf417; del buf417  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_80(c_void_p(buf421.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(mul_694.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf419.data_ptr()))
    del convolution_15
    del mul_694
    del primals_19
    del primals_232
    del primals_233
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf423 = aten.convolution_backward(buf422, add_18, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_18
    del primals_120
    buf424 = buf423[0]
    buf425 = buf423[1]
    del buf423
    buf429 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_81(c_void_p(buf424.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf429.data_ptr()))
    del primals_17
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf430 = aten.convolution_backward(buf429, mul_35, primals_119, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf429
    del mul_35
    del primals_119
    buf431 = buf430[0]
    buf433 = reinterpret_tensor(buf409, (4, 144, 1, 1), (144, 1, 576, 576), 0); del buf409  # reuse
    buf434 = reinterpret_tensor(buf433, (4, 144, 1, 1), (144, 1, 1, 1), 0); del buf433  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_82(c_void_p(buf434.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(add_15.data_ptr()), c_void_p(convolution_13.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf435 = aten.convolution_backward(buf434, mul_34, primals_117, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf434
    del mul_34
    del primals_117
    buf436 = buf435[0]
    buf439 = reinterpret_tensor(buf436, (4, 6, 1, 1), (6, 1, 1, 1), 0); del buf436  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_83(c_void_p(buf439.data_ptr()), c_void_p(convolution_12.data_ptr()))
    del convolution_12
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf440 = aten.convolution_backward(buf439, mean_2, primals_115, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_2
    del primals_115
    buf441 = buf440[0]
    buf447 = buf422; del buf422  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_84(c_void_p(buf431.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(add_15.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf447.data_ptr()))
    del primals_15
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___se_gate], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
    buf448 = aten.convolution_backward(buf447, mul_29, primals_114, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False])
    del mul_29
    del primals_114
    buf449 = buf448[0]
    buf454 = buf447; del buf447  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_85(c_void_p(buf449.data_ptr()), c_void_p(mul_731.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf454.data_ptr()))
    del primals_13
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf455 = aten.convolution_backward(buf454, add_11, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_11
    del buf454
    del primals_113
    buf456 = buf455[0]
    buf426 = reinterpret_tensor(buf439, (24, ), (1, ), 0); del buf439  # reuse
    buf427 = reinterpret_tensor(buf407, (24, ), (1, ), 0); del buf407  # reuse
    buf458 = empty((24, ), device='cpu', dtype=torch.float32)
    buf459 = empty((24, ), device='cpu', dtype=torch.float32)
    buf428 = buf427; del buf427  # reuse
    cpp_fused_add_native_batch_norm_backward_86(c_void_p(buf428.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()))
    del convolution_14
    del convolution_9
    del primals_224
    del primals_230
    del primals_231
    buf432 = buf430[1]
    del buf430
    buf437 = buf435[1]
    buf438 = buf435[2]
    del buf435
    buf442 = buf440[1]
    buf443 = buf440[2]
    del buf440
    buf444 = empty((144, ), device='cpu', dtype=torch.float32)
    buf445 = empty((144, ), device='cpu', dtype=torch.float32)
    buf446 = buf445; del buf445  # reuse
    cpp_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_87(c_void_p(buf446.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(add_15.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(buf444.data_ptr()))
    del add_15
    del buf431
    del buf441
    del convolution_11
    del convolution_13
    del primals_228
    del primals_229
    buf450 = buf448[1]
    del buf448
    buf451 = empty((144, ), device='cpu', dtype=torch.float32)
    buf452 = empty((144, ), device='cpu', dtype=torch.float32)
    buf453 = buf452; del buf452  # reuse
    cpp_fused_mul_native_batch_norm_backward_88(c_void_p(buf453.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(mul_731.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(buf451.data_ptr()))
    del buf449
    del convolution_10
    del mul_731
    del primals_226
    del primals_227
    buf457 = buf455[1]
    del buf455
    buf460 = buf459; del buf459  # reuse
    buf461 = buf424; del buf424  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_89(c_void_p(buf460.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(primals_11.data_ptr()))
    del buf456
    del primals_11
    del primals_225
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf462 = aten.convolution_backward(buf461, mul_22, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf461
    del mul_22
    del primals_112
    buf463 = buf462[0]
    buf464 = buf462[1]
    del buf462
    buf465 = empty_strided((4, 96, 1, 1), (96, 1, 384, 384), device='cpu', dtype=torch.float32)
    buf466 = reinterpret_tensor(buf465, (4, 96, 1, 1), (96, 1, 1, 1), 0); del buf465  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_90(c_void_p(buf466.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(add_9.data_ptr()), c_void_p(convolution_8.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf467 = aten.convolution_backward(buf466, mul_21, primals_110, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf466
    del mul_21
    del primals_110
    buf468 = buf467[0]
    buf469 = buf467[1]
    buf470 = buf467[2]
    del buf467
    buf471 = reinterpret_tensor(buf468, (4, 4, 1, 1), (4, 1, 1, 1), 0); del buf468  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_91(c_void_p(buf471.data_ptr()), c_void_p(convolution_7.data_ptr()))
    del convolution_7
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf472 = aten.convolution_backward(buf471, mean_1, primals_108, [4], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_1
    del primals_108
    buf473 = buf472[0]
    buf474 = buf472[1]
    buf475 = buf472[2]
    del buf472
    buf476 = empty((96, ), device='cpu', dtype=torch.float32)
    buf477 = empty((96, ), device='cpu', dtype=torch.float32)
    buf478 = buf477; del buf477  # reuse
    buf479 = buf463; del buf463  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_92(c_void_p(buf478.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(add_9.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf476.data_ptr()))
    del add_9
    del buf473
    del convolution_6
    del convolution_8
    del primals_222
    del primals_223
    del primals_9
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___se_gate], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
    buf480 = aten.convolution_backward(buf479, mul_16, primals_107, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 96, [True, True, False])
    del buf479
    del mul_16
    del primals_107
    buf481 = buf480[0]
    buf482 = buf480[1]
    del buf480
    buf483 = empty((96, ), device='cpu', dtype=torch.float32)
    buf484 = empty((96, ), device='cpu', dtype=torch.float32)
    buf485 = buf484; del buf484  # reuse
    buf486 = buf481; del buf481  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_93(c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(mul_768.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf483.data_ptr()))
    del convolution_5
    del mul_768
    del primals_220
    del primals_221
    del primals_7
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf487 = aten.convolution_backward(buf486, add_5, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_5
    del buf486
    del primals_106
    buf488 = buf487[0]
    buf489 = buf487[1]
    del buf487
    buf490 = reinterpret_tensor(buf471, (16, ), (1, ), 0); del buf471  # reuse
    buf491 = empty((16, ), device='cpu', dtype=torch.float32)
    buf492 = buf491; del buf491  # reuse
    buf493 = buf488; del buf488  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_94(c_void_p(buf492.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf490.data_ptr()))
    del convolution_4
    del primals_218
    del primals_219
    del primals_5
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf494 = aten.convolution_backward(buf493, mul_9, primals_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf493
    del mul_9
    del primals_105
    buf495 = buf494[0]
    buf496 = buf494[1]
    del buf494
    buf497 = empty_strided((4, 32, 1, 1), (32, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf498 = reinterpret_tensor(buf497, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf497  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_95(c_void_p(buf498.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(add_3.data_ptr()), c_void_p(convolution_3.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf499 = aten.convolution_backward(buf498, mul_8, primals_103, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf498
    del mul_8
    del primals_103
    buf500 = buf499[0]
    buf501 = buf499[1]
    buf502 = buf499[2]
    del buf499
    buf503 = reinterpret_tensor(buf500, (4, 8, 1, 1), (8, 1, 1, 1), 0); del buf500  # reuse
    cpp_fused_add_convolution_backward_fill_mul_sigmoid_sub_96(c_void_p(buf503.data_ptr()), c_void_p(convolution_2.data_ptr()))
    del convolution_2
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf504 = aten.convolution_backward(buf503, mean, primals_101, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean
    del primals_101
    buf505 = buf504[0]
    buf506 = buf504[1]
    buf507 = buf504[2]
    del buf504
    buf508 = reinterpret_tensor(buf503, (32, ), (1, ), 0); del buf503  # reuse
    buf509 = empty((32, ), device='cpu', dtype=torch.float32)
    buf510 = buf509; del buf509  # reuse
    buf511 = buf495; del buf495  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_97(c_void_p(buf510.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(add_3.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(primals_216.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf508.data_ptr()))
    del add_3
    del buf505
    del convolution_1
    del convolution_3
    del primals_216
    del primals_217
    del primals_3
    # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___se_gate], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
    buf512 = aten.convolution_backward(buf511, mul_3, primals_100, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf511
    del mul_3
    del primals_100
    buf513 = buf512[0]
    buf514 = buf512[1]
    del buf512
    buf515 = empty((32, ), device='cpu', dtype=torch.float32)
    buf516 = empty((32, ), device='cpu', dtype=torch.float32)
    buf517 = buf516; del buf516  # reuse
    buf518 = buf513; del buf513  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_98(c_void_p(buf517.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(mul_805.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf515.data_ptr()))
    del convolution
    del mul_805
    del primals_1
    del primals_214
    del primals_215
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf519 = aten.convolution_backward(buf518, primals_312, primals_99, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf518
    del primals_312
    del primals_99
    buf520 = buf519[1]
    return (buf517, buf515, buf510, buf508, buf492, buf490, buf485, buf483, buf478, buf476, buf460, buf458, buf453, buf451, buf446, buf444, buf428, buf426, buf421, buf419, buf414, buf412, buf396, buf394, buf389, buf387, buf382, buf380, buf364, buf362, buf357, buf355, buf350, buf348, buf332, buf330, buf325, buf323, buf318, buf316, buf300, buf298, buf293, buf291, buf286, buf284, buf268, buf266, buf261, buf259, buf254, buf252, buf236, buf234, buf229, buf227, buf222, buf220, buf204, buf202, buf197, buf195, buf190, buf188, buf172, buf170, buf165, buf163, buf158, buf156, buf140, buf138, buf133, buf131, buf126, buf124, buf108, buf106, buf101, buf99, buf94, buf92, buf76, buf74, buf69, buf67, buf62, buf60, buf44, buf42, buf37, buf35, buf30, buf28, buf12, buf10, buf5, buf3, buf520, buf514, buf506, buf507, buf501, buf502, buf496, buf489, buf482, buf474, buf475, buf469, buf470, buf464, buf457, buf450, buf442, buf443, buf437, buf438, buf432, buf425, buf418, buf410, buf411, buf405, buf406, buf400, buf393, buf386, buf378, buf379, buf373, buf374, buf368, buf361, buf354, buf346, buf347, buf341, buf342, buf336, buf329, buf322, buf314, buf315, buf309, buf310, buf304, buf297, buf290, buf282, buf283, buf277, buf278, buf272, buf265, buf258, buf250, buf251, buf245, buf246, buf240, buf233, buf226, buf218, buf219, buf213, buf214, buf208, buf201, buf194, buf186, buf187, buf181, buf182, buf176, buf169, buf162, buf154, buf155, buf149, buf150, buf144, buf137, buf130, buf122, buf123, buf117, buf118, buf112, buf105, buf98, buf90, buf91, buf85, buf86, buf80, buf73, buf66, buf58, buf59, buf53, buf54, buf48, buf41, buf34, buf26, buf27, buf21, buf22, buf16, buf9, reinterpret_tensor(buf1, (1000, 1280), (1280, 1), 0), buf2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


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
    primals_53 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((8, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((4, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((96, 4, 1, 1), (4, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((24, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((6, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((144, 6, 1, 1), (6, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((40, 144, 1, 1), (144, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((10, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((240, 10, 1, 1), (10, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((20, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((480, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((28, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((672, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
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
    primals_214 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((144, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    mul_3 = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    add_3 = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    mean = rand_strided((4, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((4, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    mul_8 = rand_strided((4, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((4, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    mul_9 = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    add_5 = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((4, 96, 112, 112), (1204224, 1, 10752, 96), device='cpu', dtype=torch.float32)
    mul_16 = rand_strided((4, 96, 112, 112), (1204224, 1, 10752, 96), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((4, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    add_9 = rand_strided((4, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    mean_1 = rand_strided((4, 96, 1, 1), (96, 1, 96, 96), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((4, 4, 1, 1), (4, 1, 4, 4), device='cpu', dtype=torch.float32)
    mul_21 = rand_strided((4, 4, 1, 1), (4, 1, 4, 4), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((4, 96, 1, 1), (96, 1, 96, 96), device='cpu', dtype=torch.float32)
    mul_22 = rand_strided((4, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    add_11 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    mul_29 = rand_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    add_15 = rand_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    mean_2 = rand_strided((4, 144, 1, 1), (144, 1, 144, 144), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((4, 6, 1, 1), (6, 1, 6, 6), device='cpu', dtype=torch.float32)
    mul_34 = rand_strided((4, 6, 1, 1), (6, 1, 6, 6), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((4, 144, 1, 1), (144, 1, 144, 144), device='cpu', dtype=torch.float32)
    mul_35 = rand_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    add_18 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    mul_42 = rand_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((4, 144, 28, 28), (112896, 1, 4032, 144), device='cpu', dtype=torch.float32)
    add_22 = rand_strided((4, 144, 28, 28), (112896, 1, 4032, 144), device='cpu', dtype=torch.float32)
    mean_3 = rand_strided((4, 144, 1, 1), (144, 1, 144, 144), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((4, 6, 1, 1), (6, 1, 6, 6), device='cpu', dtype=torch.float32)
    mul_47 = rand_strided((4, 6, 1, 1), (6, 1, 6, 6), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((4, 144, 1, 1), (144, 1, 144, 144), device='cpu', dtype=torch.float32)
    mul_48 = rand_strided((4, 144, 28, 28), (112896, 1, 4032, 144), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    add_24 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    mul_55 = rand_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    add_28 = rand_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    mean_4 = rand_strided((4, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((4, 10, 1, 1), (10, 1, 10, 10), device='cpu', dtype=torch.float32)
    mul_60 = rand_strided((4, 10, 1, 1), (10, 1, 10, 10), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((4, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    mul_61 = rand_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    add_31 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    mul_68 = rand_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    add_35 = rand_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    mean_5 = rand_strided((4, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((4, 10, 1, 1), (10, 1, 10, 10), device='cpu', dtype=torch.float32)
    mul_73 = rand_strided((4, 10, 1, 1), (10, 1, 10, 10), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((4, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    mul_74 = rand_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    add_37 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    mul_81 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    add_41 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    mean_6 = rand_strided((4, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((4, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    mul_86 = rand_strided((4, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((4, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    mul_87 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    add_44 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    mul_94 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    add_48 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    mean_7 = rand_strided((4, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((4, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    mul_99 = rand_strided((4, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((4, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    mul_100 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    add_51 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    mul_107 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    add_55 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    mean_8 = rand_strided((4, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((4, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    mul_112 = rand_strided((4, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((4, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    mul_113 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    add_57 = rand_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    mul_120 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    add_61 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    mean_9 = rand_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((4, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    mul_125 = rand_strided((4, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    mul_126 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    add_64 = rand_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    mul_133 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    add_68 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    mean_10 = rand_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((4, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    mul_138 = rand_strided((4, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    convolution_53 = rand_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    mul_139 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    convolution_54 = rand_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    add_71 = rand_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    mul_146 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    convolution_56 = rand_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    add_75 = rand_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    mean_11 = rand_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    convolution_57 = rand_strided((4, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    mul_151 = rand_strided((4, 28, 1, 1), (28, 1, 28, 28), device='cpu', dtype=torch.float32)
    convolution_58 = rand_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    mul_152 = rand_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    convolution_59 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    add_77 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    convolution_60 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    mul_159 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_61 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    add_81 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    mean_12 = rand_strided((4, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    convolution_62 = rand_strided((4, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    mul_164 = rand_strided((4, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    convolution_63 = rand_strided((4, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    mul_165 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_64 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    add_84 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    convolution_65 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    mul_172 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_66 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    add_88 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    mean_13 = rand_strided((4, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    convolution_67 = rand_strided((4, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    mul_177 = rand_strided((4, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    convolution_68 = rand_strided((4, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    mul_178 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_69 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    add_91 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    convolution_70 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    mul_185 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_71 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    add_95 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    mean_14 = rand_strided((4, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    convolution_72 = rand_strided((4, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    mul_190 = rand_strided((4, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    convolution_73 = rand_strided((4, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    mul_191 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_74 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    add_98 = rand_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cpu', dtype=torch.float32)
    convolution_75 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    mul_198 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_76 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    add_102 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    mean_15 = rand_strided((4, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    convolution_77 = rand_strided((4, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    mul_203 = rand_strided((4, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    convolution_78 = rand_strided((4, 1152, 1, 1), (1152, 1, 1152, 1152), device='cpu', dtype=torch.float32)
    mul_204 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    convolution_79 = rand_strided((4, 320, 7, 7), (15680, 1, 2240, 320), device='cpu', dtype=torch.float32)
    add_104 = rand_strided((4, 320, 7, 7), (15680, 1, 2240, 320), device='cpu', dtype=torch.float32)
    convolution_80 = rand_strided((4, 1280, 7, 7), (62720, 1, 8960, 1280), device='cpu', dtype=torch.float32)
    view = rand_strided((4, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    mul_213 = rand_strided((4, 1280, 7, 7), (62720, 1, 8960, 1280), device='cpu', dtype=torch.float32)
    mul_250 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    mul_287 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    mul_324 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    mul_361 = rand_strided((4, 1152, 7, 7), (56448, 1, 8064, 1152), device='cpu', dtype=torch.float32)
    mul_398 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    mul_435 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    mul_472 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    mul_509 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    mul_546 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    mul_583 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    mul_620 = rand_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    mul_657 = rand_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    mul_694 = rand_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    mul_731 = rand_strided((4, 144, 56, 56), (451584, 1, 8064, 144), device='cpu', dtype=torch.float32)
    mul_768 = rand_strided((4, 96, 112, 112), (1204224, 1, 10752, 96), device='cpu', dtype=torch.float32)
    mul_805 = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_100, primals_101, primals_103, primals_105, primals_106, primals_107, primals_108, primals_110, primals_112, primals_113, primals_114, primals_115, primals_117, primals_119, primals_120, primals_121, primals_122, primals_124, primals_126, primals_127, primals_128, primals_129, primals_131, primals_133, primals_134, primals_135, primals_136, primals_138, primals_140, primals_141, primals_142, primals_143, primals_145, primals_147, primals_148, primals_149, primals_150, primals_152, primals_154, primals_155, primals_156, primals_157, primals_159, primals_161, primals_162, primals_163, primals_164, primals_166, primals_168, primals_169, primals_170, primals_171, primals_173, primals_175, primals_176, primals_177, primals_178, primals_180, primals_182, primals_183, primals_184, primals_185, primals_187, primals_189, primals_190, primals_191, primals_192, primals_194, primals_196, primals_197, primals_198, primals_199, primals_201, primals_203, primals_204, primals_205, primals_206, primals_208, primals_210, primals_211, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, convolution, mul_3, convolution_1, add_3, mean, convolution_2, mul_8, convolution_3, mul_9, convolution_4, add_5, convolution_5, mul_16, convolution_6, add_9, mean_1, convolution_7, mul_21, convolution_8, mul_22, convolution_9, add_11, convolution_10, mul_29, convolution_11, add_15, mean_2, convolution_12, mul_34, convolution_13, mul_35, convolution_14, add_18, convolution_15, mul_42, convolution_16, add_22, mean_3, convolution_17, mul_47, convolution_18, mul_48, convolution_19, add_24, convolution_20, mul_55, convolution_21, add_28, mean_4, convolution_22, mul_60, convolution_23, mul_61, convolution_24, add_31, convolution_25, mul_68, convolution_26, add_35, mean_5, convolution_27, mul_73, convolution_28, mul_74, convolution_29, add_37, convolution_30, mul_81, convolution_31, add_41, mean_6, convolution_32, mul_86, convolution_33, mul_87, convolution_34, add_44, convolution_35, mul_94, convolution_36, add_48, mean_7, convolution_37, mul_99, convolution_38, mul_100, convolution_39, add_51, convolution_40, mul_107, convolution_41, add_55, mean_8, convolution_42, mul_112, convolution_43, mul_113, convolution_44, add_57, convolution_45, mul_120, convolution_46, add_61, mean_9, convolution_47, mul_125, convolution_48, mul_126, convolution_49, add_64, convolution_50, mul_133, convolution_51, add_68, mean_10, convolution_52, mul_138, convolution_53, mul_139, convolution_54, add_71, convolution_55, mul_146, convolution_56, add_75, mean_11, convolution_57, mul_151, convolution_58, mul_152, convolution_59, add_77, convolution_60, mul_159, convolution_61, add_81, mean_12, convolution_62, mul_164, convolution_63, mul_165, convolution_64, add_84, convolution_65, mul_172, convolution_66, add_88, mean_13, convolution_67, mul_177, convolution_68, mul_178, convolution_69, add_91, convolution_70, mul_185, convolution_71, add_95, mean_14, convolution_72, mul_190, convolution_73, mul_191, convolution_74, add_98, convolution_75, mul_198, convolution_76, add_102, mean_15, convolution_77, mul_203, convolution_78, mul_204, convolution_79, add_104, convolution_80, view, permute_1, mul_213, mul_250, mul_287, mul_324, mul_361, mul_398, mul_435, mul_472, mul_509, mul_546, mul_583, mul_620, mul_657, mul_694, mul_731, mul_768, mul_805, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('timm_efficientnet', benchmark_compiled_module)
