
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


cpp_fused_add_convolution_backward_div_fill_mul_sigmoid_sub_sum_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (2304L*x2) + (112896L*x0)), static_cast<long>(2304L), tmp3, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (2304L*x2) + (112896L*x0)), static_cast<long>(2304L), tmp3, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (2304L*x2) + (112896L*x0)), static_cast<long>(2304L), tmp3, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x1 + x1_inner + (2304L*x0))];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(tmp3 + static_cast<long>(8L*x1_inner));
                            auto tmp1 = static_cast<float>(49.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = decltype(tmp4)(1)/(decltype(tmp4)(1) + tmp4.neg().exp());
                            auto tmp6 = static_cast<float>(1.0);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp7 - tmp5;
                            auto tmp9 = tmp4 * tmp8;
                            auto tmp10 = tmp9 + tmp7;
                            auto tmp11 = tmp5 * tmp10;
                            auto tmp12 = at::vec::Vectorized<float>(tmp2);
                            auto tmp13 = tmp12 * tmp11;
                            tmp13.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (112896L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2304L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2304L*x2) + (112896L*x0)));
                        auto tmp1 = static_cast<float>(49.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        auto tmp5 = decltype(tmp4)(1)/(decltype(tmp4)(1) + tmp4.neg().exp());
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp7 - tmp5;
                        auto tmp9 = tmp4 * tmp8;
                        auto tmp10 = tmp9 + tmp7;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp3 * tmp11;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp12.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (112896L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_1 = async_compile.cpp('''
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
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.04562504637317021);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.04562504637317021);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1536L*x2) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp4 = static_cast<float>(2.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp8 = tmp6 * tmp7;
                            tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_sigmoid_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp1 = static_cast<float>(0.2);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = decltype(tmp7)(1)/(decltype(tmp7)(1) + tmp7.neg().exp());
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = static_cast<float>(49.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        auto tmp14 = tmp9 + tmp13;
                        tmp14.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.09125009274634042);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0026041666666666665);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.09125009274634042);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(150528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07450538873672485);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.001736111111111111);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.07450538873672485);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(150528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07450538873672485);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.001736111111111111);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.07450538873672485);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(150528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_7 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.04562504637317021);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.04562504637317021);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1536L*x2) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (1536L*x2) + (75264L*x0)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (1536L*x2) + (75264L*x0)));
                            auto tmp2 = static_cast<float>(0.9622504486493761);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp8 = static_cast<float>(0.2);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 * tmp9;
                            auto tmp11 = static_cast<float>(2.0);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 * tmp12;
                            auto tmp15 = tmp13 * tmp14;
                            tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_sigmoid_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp2 = static_cast<float>(0.9622504486493761);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp8 = static_cast<float>(0.2);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = static_cast<float>(2.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 * tmp12;
                        auto tmp15 = decltype(tmp14)(1)/(decltype(tmp14)(1) + tmp14.neg().exp());
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp18 = static_cast<float>(49.0);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 / tmp19;
                        auto tmp21 = tmp16 + tmp20;
                        tmp21.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.09125009274634042);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0026041666666666665);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.09125009274634042);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(150528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07450538873672485);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.001736111111111111);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.07450538873672485);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(150528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07450538873672485);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.001736111111111111);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.07450538873672485);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(150528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
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
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr2;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.04562504637317021);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.04562504637317021);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.9622504486493761);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp6 = tmp4 * tmp5;
                auto tmp7 = tmp0 + tmp6;
                auto tmp9 = static_cast<float>(0.9805806756909201);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp14 = tmp7 + tmp13;
                tmp14.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1536L*x2) + (75264L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp4 = static_cast<float>(2.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp8 = tmp6 * tmp7;
                            tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_sigmoid_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp1 = static_cast<float>(0.2);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = decltype(tmp7)(1)/(decltype(tmp7)(1) + tmp7.neg().exp());
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = static_cast<float>(49.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        auto tmp14 = tmp9 + tmp13;
                        tmp14.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (75264L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.09125009274634042);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0026041666666666665);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.09125009274634042);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(150528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07450538873672485);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.001736111111111111);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.07450538873672485);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(150528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07450538873672485);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.001736111111111111);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.07450538873672485);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_mul_native_batch_norm_backward_view_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.04562504637317021);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.04562504637317021);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_avg_pool2d_backward_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_20 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.04562504637317021);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.04562504637317021);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr5[static_cast<long>(x1 + (1536L*x3) + (21504L*x2) + (301056L*x0))];
                                auto tmp1 = in_ptr6[static_cast<long>(x1 + (1536L*(std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x3, 2L))))))) + (1536L*(((std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x3, 2L)))))) >= 0L) ? 0L : 7L)) + (10752L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x2, 2L))))))) + (10752L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x2, 2L)))))) >= 0L) ? 0L : 7L)) + (75264L*x0))];
                                auto tmp15 = in_ptr7[static_cast<long>(x1 + (1536L*x3) + (21504L*x2) + (301056L*x0))];
                                auto tmp21 = in_ptr8[static_cast<long>(x1 + (1536L*x3) + (21504L*x2) + (301056L*x0))];
                                auto tmp2 = tmp1 / 4;
                                auto tmp3 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                                auto tmp4 = c10::convert<int>(std::min(7L, 1L + (c10::div_floor_integer(x2, 2L))));
                                auto tmp5 = tmp3 < tmp4;
                                auto tmp6 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x3, 2L)));
                                auto tmp7 = c10::convert<int>(std::min(7L, 1L + (c10::div_floor_integer(x3, 2L))));
                                auto tmp8 = tmp6 < tmp7;
                                auto tmp9 = tmp5 & tmp8;
                                auto tmp10 = static_cast<float>(0.0);
                                auto tmp11 = tmp9 ? tmp2 : tmp10;
                                auto tmp12 = decltype(tmp0)(tmp0 + tmp11);
                                auto tmp13 = static_cast<float>(0.8980265101338745);
                                auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                                auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                                auto tmp17 = static_cast<float>(0.2);
                                auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                                auto tmp19 = static_cast<float>(2.0);
                                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                                auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                                tmp_acc0 = tmp_acc0 + tmp22;
                            }
                        }
                        out_ptr3[static_cast<long>(x1 + (1536L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
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


cpp_fused_add_avg_pool2d_backward_convolution_backward_div_mul_sigmoid_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1536L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (1536L*x2) + (21504L*x1) + (301056L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (1536L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x2, 2L))))))) + (1536L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x2, 2L)))))) >= 0L) ? 0L : 7L)) + (10752L*(std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x1, 2L))))))) + (10752L*(((std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x1, 2L)))))) >= 0L) ? 0L : 7L)) + (75264L*x0))];
                            auto tmp15 = in_ptr2[static_cast<long>(x3 + (1536L*x2) + (21504L*x1) + (301056L*x0))];
                            auto tmp21 = in_ptr3[static_cast<long>(x3 + (1536L*x0))];
                            auto tmp24 = in_ptr4[static_cast<long>(x3 + (1536L*x0))];
                            auto tmp2 = tmp1 / 4;
                            auto tmp3 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x1, 2L)));
                            auto tmp4 = c10::convert<int>(std::min(7L, 1L + (c10::div_floor_integer(x1, 2L))));
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                            auto tmp7 = c10::convert<int>(std::min(7L, 1L + (c10::div_floor_integer(x2, 2L))));
                            auto tmp8 = tmp6 < tmp7;
                            auto tmp9 = tmp5 & tmp8;
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp9 ? tmp2 : tmp10;
                            auto tmp12 = decltype(tmp0)(tmp0 + tmp11);
                            auto tmp13 = static_cast<float>(0.8980265101338745);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp17 = static_cast<float>(0.2);
                            auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                            auto tmp19 = static_cast<float>(2.0);
                            auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                            auto tmp22 = decltype(tmp21)(1) / (decltype(tmp21)(1) + std::exp(-tmp21));
                            auto tmp23 = decltype(tmp20)(tmp20 * tmp22);
                            auto tmp25 = static_cast<float>(196.0);
                            auto tmp26 = tmp24 / tmp25;
                            auto tmp27 = decltype(tmp23)(tmp23 + tmp26);
                            out_ptr0[static_cast<long>(x3 + (1536L*x2) + (21504L*x1) + (301056L*x0))] = tmp27;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.09125009274634042);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0026041666666666665);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.09125009274634042);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07450538873672485);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.001736111111111111);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.07450538873672485);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07450538873672485);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.001736111111111111);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.07450538873672485);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_avg_pool2d_backward_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
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
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr2;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.04562504637317021);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.04562504637317021);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
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
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1536L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr1[static_cast<long>(x3 + (1536L*x2) + (21504L*x1) + (301056L*x0))];
                            auto tmp1 = in_ptr5[static_cast<long>(x3 + (1536L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x2, 2L))))))) + (1536L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x2, 2L)))))) >= 0L) ? 0L : 7L)) + (10752L*(std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x1, 2L))))))) + (10752L*(((std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(7L, 1L + (c10::div_floor_integer(x1, 2L)))))) >= 0L) ? 0L : 7L)) + (75264L*x0))];
                            auto tmp15 = in_ptr6[static_cast<long>(x3 + (1536L*x2) + (21504L*x1) + (301056L*x0))];
                            auto tmp17 = in_ptr7[static_cast<long>(x3 + (1536L*x2) + (21504L*x1) + (301056L*x0))];
                            auto tmp20 = in_ptr8[static_cast<long>(x3 + (1536L*x2) + (21504L*x1) + (301056L*x0))];
                            auto tmp2 = tmp1 / 4;
                            auto tmp3 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x1, 2L)));
                            auto tmp4 = c10::convert<int>(std::min(7L, 1L + (c10::div_floor_integer(x1, 2L))));
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                            auto tmp7 = c10::convert<int>(std::min(7L, 1L + (c10::div_floor_integer(x2, 2L))));
                            auto tmp8 = tmp6 < tmp7;
                            auto tmp9 = tmp5 & tmp8;
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp9 ? tmp2 : tmp10;
                            auto tmp12 = decltype(tmp0)(tmp0 + tmp11);
                            auto tmp13 = static_cast<float>(0.8980265101338745);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp18 = static_cast<float>(0.9128709291752768);
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = decltype(tmp16)(tmp16 + tmp21);
                            in_out_ptr1[static_cast<long>(x3 + (1536L*x2) + (21504L*x1) + (301056L*x0))] = tmp22;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp4 = static_cast<float>(2.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp8 = tmp6 * tmp7;
                            tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_sigmoid_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = static_cast<float>(0.2);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = decltype(tmp7)(1)/(decltype(tmp7)(1) + tmp7.neg().exp());
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = static_cast<float>(196.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        auto tmp14 = tmp9 + tmp13;
                        tmp14.store(out_ptr0 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.09125009274634042);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0026041666666666665);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.09125009274634042);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07450538873672485);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.001736111111111111);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.07450538873672485);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07450538873672485);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.001736111111111111);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.07450538873672485);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_32 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.04562504637317021);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.04562504637317021);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            auto tmp2 = static_cast<float>(0.9284766908852592);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp8 = static_cast<float>(0.2);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 * tmp9;
                            auto tmp11 = static_cast<float>(2.0);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 * tmp12;
                            auto tmp15 = tmp13 * tmp14;
                            tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_sigmoid_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp2 = static_cast<float>(0.9284766908852592);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp8 = static_cast<float>(0.2);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = static_cast<float>(2.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 * tmp12;
                        auto tmp15 = decltype(tmp14)(1)/(decltype(tmp14)(1) + tmp14.neg().exp());
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp18 = static_cast<float>(196.0);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 / tmp19;
                        auto tmp21 = tmp16 + tmp20;
                        tmp21.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.09125009274634042);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0026041666666666665);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.09125009274634042);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07450538873672485);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.001736111111111111);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.07450538873672485);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07450538873672485);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.001736111111111111);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.07450538873672485);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
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
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr2;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.04562504637317021);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.04562504637317021);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.9284766908852592);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp6 = tmp4 * tmp5;
                auto tmp7 = tmp0 + tmp6;
                auto tmp9 = static_cast<float>(0.9449111825230679);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp14 = tmp7 + tmp13;
                tmp14.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp4 = static_cast<float>(2.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp8 = tmp6 * tmp7;
                            tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_sigmoid_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp1 = static_cast<float>(0.2);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = decltype(tmp7)(1)/(decltype(tmp7)(1) + tmp7.neg().exp());
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = static_cast<float>(196.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        auto tmp14 = tmp9 + tmp13;
                        tmp14.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.09125009274634042);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0026041666666666665);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.09125009274634042);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07450538873672485);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.001736111111111111);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.07450538873672485);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07450538873672485);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.001736111111111111);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.07450538873672485);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_44 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.04562504637317021);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.04562504637317021);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            auto tmp2 = static_cast<float>(0.9622504486493761);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp8 = static_cast<float>(0.2);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 * tmp9;
                            auto tmp11 = static_cast<float>(2.0);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 * tmp12;
                            auto tmp15 = tmp13 * tmp14;
                            tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_sigmoid_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp2 = static_cast<float>(0.9622504486493761);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp8 = static_cast<float>(0.2);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = static_cast<float>(2.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 * tmp12;
                        auto tmp15 = decltype(tmp14)(1)/(decltype(tmp14)(1) + tmp14.neg().exp());
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp18 = static_cast<float>(196.0);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 / tmp19;
                        auto tmp21 = tmp16 + tmp20;
                        tmp21.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.09125009274634042);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0026041666666666665);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.09125009274634042);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07450538873672485);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.001736111111111111);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.07450538873672485);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07450538873672485);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.001736111111111111);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.07450538873672485);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
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
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr2;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.04562504637317021);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006510416666666666);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.04562504637317021);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.9622504486493761);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp6 = tmp4 * tmp5;
                auto tmp7 = tmp0 + tmp6;
                auto tmp9 = static_cast<float>(0.9805806756909201);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp14 = tmp7 + tmp13;
                tmp14.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp4 = static_cast<float>(2.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp8 = tmp6 * tmp7;
                            tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_sigmoid_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp1 = static_cast<float>(0.2);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = decltype(tmp7)(1)/(decltype(tmp7)(1) + tmp7.neg().exp());
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = static_cast<float>(196.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        auto tmp14 = tmp9 + tmp13;
                        tmp14.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (301056L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.09125009274634042);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.0026041666666666665);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.09125009274634042);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07450538873672485);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.001736111111111111);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.07450538873672485);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07450538873672485);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.001736111111111111);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp8 * tmp8;
                        auto tmp10 = tmp7 * tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp12 = tmp0 - tmp11;
                        auto tmp14 = tmp13 * tmp6;
                        auto tmp15 = tmp12 - tmp14;
                        auto tmp17 = static_cast<float>(0.07450538873672485);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp8 * tmp19;
                        auto tmp21 = tmp15 * tmp20;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_mul_native_batch_norm_backward_view_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07902489841601695);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.001953125);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.07902489841601695);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_avg_pool2d_backward_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_57 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.07902489841601695);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.001953125);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.07902489841601695);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(28L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr5[static_cast<long>(x1 + (512L*x3) + (14336L*x2) + (401408L*x0))];
                                auto tmp1 = in_ptr6[static_cast<long>(x1 + (512L*(std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x3, 2L))))))) + (512L*(((std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x3, 2L)))))) >= 0L) ? 0L : 14L)) + (7168L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x2, 2L))))))) + (7168L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x2, 2L)))))) >= 0L) ? 0L : 14L)) + (100352L*x0))];
                                auto tmp15 = in_ptr7[static_cast<long>(x1 + (512L*x3) + (14336L*x2) + (401408L*x0))];
                                auto tmp21 = in_ptr8[static_cast<long>(x1 + (512L*x3) + (14336L*x2) + (401408L*x0))];
                                auto tmp2 = tmp1 / 4;
                                auto tmp3 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                                auto tmp4 = c10::convert<int>(std::min(14L, 1L + (c10::div_floor_integer(x2, 2L))));
                                auto tmp5 = tmp3 < tmp4;
                                auto tmp6 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x3, 2L)));
                                auto tmp7 = c10::convert<int>(std::min(14L, 1L + (c10::div_floor_integer(x3, 2L))));
                                auto tmp8 = tmp6 < tmp7;
                                auto tmp9 = tmp5 & tmp8;
                                auto tmp10 = static_cast<float>(0.0);
                                auto tmp11 = tmp9 ? tmp2 : tmp10;
                                auto tmp12 = decltype(tmp0)(tmp0 + tmp11);
                                auto tmp13 = static_cast<float>(0.9622504486493761);
                                auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                                auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                                auto tmp17 = static_cast<float>(0.2);
                                auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                                auto tmp19 = static_cast<float>(2.0);
                                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                                auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                                tmp_acc0 = tmp_acc0 + tmp22;
                            }
                        }
                        out_ptr3[static_cast<long>(x1 + (512L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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


cpp_fused_add_avg_pool2d_backward_convolution_backward_div_mul_sigmoid_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (512L*x2) + (14336L*x1) + (401408L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (512L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x2, 2L))))))) + (512L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x2, 2L)))))) >= 0L) ? 0L : 14L)) + (7168L*(std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x1, 2L))))))) + (7168L*(((std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x1, 2L)))))) >= 0L) ? 0L : 14L)) + (100352L*x0))];
                            auto tmp15 = in_ptr2[static_cast<long>(x3 + (512L*x2) + (14336L*x1) + (401408L*x0))];
                            auto tmp21 = in_ptr3[static_cast<long>(x3 + (512L*x0))];
                            auto tmp24 = in_ptr4[static_cast<long>(x3 + (512L*x0))];
                            auto tmp2 = tmp1 / 4;
                            auto tmp3 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x1, 2L)));
                            auto tmp4 = c10::convert<int>(std::min(14L, 1L + (c10::div_floor_integer(x1, 2L))));
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                            auto tmp7 = c10::convert<int>(std::min(14L, 1L + (c10::div_floor_integer(x2, 2L))));
                            auto tmp8 = tmp6 < tmp7;
                            auto tmp9 = tmp5 & tmp8;
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp9 ? tmp2 : tmp10;
                            auto tmp12 = decltype(tmp0)(tmp0 + tmp11);
                            auto tmp13 = static_cast<float>(0.9622504486493761);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp17 = static_cast<float>(0.2);
                            auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                            auto tmp19 = static_cast<float>(2.0);
                            auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                            auto tmp22 = decltype(tmp21)(1) / (decltype(tmp21)(1) + std::exp(-tmp21));
                            auto tmp23 = decltype(tmp20)(tmp20 * tmp22);
                            auto tmp25 = static_cast<float>(784.0);
                            auto tmp26 = tmp24 / tmp25;
                            auto tmp27 = decltype(tmp23)(tmp23 + tmp26);
                            out_ptr0[static_cast<long>(x3 + (512L*x2) + (14336L*x1) + (401408L*x0))] = tmp27;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.1580497968320339);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = in_ptr2[static_cast<long>(x0)];
                auto tmp5 = out_ptr1[static_cast<long>(x0)];
                auto tmp8 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = out_ptr0[static_cast<long>(x0)];
                auto tmp18 = in_ptr4[static_cast<long>(x0)];
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 - tmp3;
                auto tmp6 = static_cast<float>(0.0078125);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp4 * tmp11;
                auto tmp13 = tmp0 - tmp12;
                auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp13 - tmp16;
                auto tmp19 = static_cast<float>(0.1580497968320339);
                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp17 * tmp22;
                tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.07450538873672485);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.001736111111111111);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = static_cast<float>(0.07450538873672485);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp8 * tmp19;
                    auto tmp21 = tmp15 * tmp20;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.07450538873672485);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.001736111111111111);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = static_cast<float>(0.07450538873672485);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp8 * tmp19;
                    auto tmp21 = tmp15 * tmp20;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_avg_pool2d_backward_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
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
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr2;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.07902489841601695);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp2 = in_ptr2[static_cast<long>(x0)];
                auto tmp5 = out_ptr1[static_cast<long>(x0)];
                auto tmp8 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = out_ptr0[static_cast<long>(x0)];
                auto tmp18 = in_ptr4[static_cast<long>(x0)];
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 - tmp3;
                auto tmp6 = static_cast<float>(0.001953125);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp4 * tmp11;
                auto tmp13 = tmp0 - tmp12;
                auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp13 - tmp16;
                auto tmp19 = static_cast<float>(0.07902489841601695);
                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp17 * tmp22;
                tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr1[static_cast<long>(x3 + (512L*x2) + (14336L*x1) + (401408L*x0))];
                            auto tmp1 = in_ptr5[static_cast<long>(x3 + (512L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x2, 2L))))))) + (512L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x2, 2L)))))) >= 0L) ? 0L : 14L)) + (7168L*(std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x1, 2L))))))) + (7168L*(((std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(14L, 1L + (c10::div_floor_integer(x1, 2L)))))) >= 0L) ? 0L : 14L)) + (100352L*x0))];
                            auto tmp15 = in_ptr6[static_cast<long>(x3 + (512L*x2) + (14336L*x1) + (401408L*x0))];
                            auto tmp17 = in_ptr7[static_cast<long>(x3 + (512L*x2) + (14336L*x1) + (401408L*x0))];
                            auto tmp20 = in_ptr8[static_cast<long>(x3 + (512L*x2) + (14336L*x1) + (401408L*x0))];
                            auto tmp2 = tmp1 / 4;
                            auto tmp3 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x1, 2L)));
                            auto tmp4 = c10::convert<int>(std::min(14L, 1L + (c10::div_floor_integer(x1, 2L))));
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                            auto tmp7 = c10::convert<int>(std::min(14L, 1L + (c10::div_floor_integer(x2, 2L))));
                            auto tmp8 = tmp6 < tmp7;
                            auto tmp9 = tmp5 & tmp8;
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp9 ? tmp2 : tmp10;
                            auto tmp12 = decltype(tmp0)(tmp0 + tmp11);
                            auto tmp13 = static_cast<float>(0.9622504486493761);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp18 = static_cast<float>(0.9805806756909201);
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                            auto tmp22 = decltype(tmp16)(tmp16 + tmp21);
                            in_out_ptr1[static_cast<long>(x3 + (512L*x2) + (14336L*x1) + (401408L*x0))] = tmp22;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (512L*x2) + (401408L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (512L*x2) + (401408L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp4 = static_cast<float>(2.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp8 = tmp6 * tmp7;
                            tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_mul_sigmoid_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (401408L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = static_cast<float>(0.2);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = decltype(tmp7)(1)/(decltype(tmp7)(1) + tmp7.neg().exp());
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = static_cast<float>(784.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        auto tmp14 = tmp9 + tmp13;
                        tmp14.store(out_ptr0 + static_cast<long>(x1 + (512L*x2) + (401408L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.1580497968320339);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = in_ptr2[static_cast<long>(x0)];
                auto tmp5 = out_ptr1[static_cast<long>(x0)];
                auto tmp8 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = out_ptr0[static_cast<long>(x0)];
                auto tmp18 = in_ptr4[static_cast<long>(x0)];
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 - tmp3;
                auto tmp6 = static_cast<float>(0.0078125);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp4 * tmp11;
                auto tmp13 = tmp0 - tmp12;
                auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp13 - tmp16;
                auto tmp19 = static_cast<float>(0.1580497968320339);
                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp17 * tmp22;
                tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.07450538873672485);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.001736111111111111);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = static_cast<float>(0.07450538873672485);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp8 * tmp19;
                    auto tmp21 = tmp15 * tmp20;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.07450538873672485);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.001736111111111111);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = static_cast<float>(0.07450538873672485);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp8 * tmp19;
                    auto tmp21 = tmp15 * tmp20;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_mul_native_batch_norm_backward_view_69 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.11175808310508728);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = in_ptr2[static_cast<long>(x0)];
                auto tmp5 = out_ptr1[static_cast<long>(x0)];
                auto tmp8 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = out_ptr0[static_cast<long>(x0)];
                auto tmp18 = in_ptr4[static_cast<long>(x0)];
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 - tmp3;
                auto tmp6 = static_cast<float>(0.00390625);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp4 * tmp11;
                auto tmp13 = tmp0 - tmp12;
                auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp13 - tmp16;
                auto tmp19 = static_cast<float>(0.11175808310508728);
                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp17 * tmp22;
                tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_avg_pool2d_backward_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr2;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<float>(0.11175808310508728);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 * tmp4;
                    tmp5.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = in_ptr3[static_cast<long>(x0)];
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp18 = in_ptr4[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp6 = static_cast<float>(0.00390625);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp0 - tmp12;
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp13 - tmp16;
                    auto tmp19 = static_cast<float>(0.11175808310508728);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr1[static_cast<long>(x3 + (256L*x2) + (14336L*x1) + (802816L*x0))];
                            auto tmp1 = in_ptr5[static_cast<long>(x3 + (256L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer(x2, 2L))))))) + (256L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer(x2, 2L)))))) >= 0L) ? 0L : 28L)) + (7168L*(std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer(x1, 2L))))))) + (7168L*(((std::min(std::max(0L, c10::div_floor_integer(x1, 2L)), (-1L) + (std::min(28L, 1L + (c10::div_floor_integer(x1, 2L)))))) >= 0L) ? 0L : 28L)) + (200704L*x0))];
                            auto tmp15 = in_ptr6[static_cast<long>(x3 + (256L*x2) + (14336L*x1) + (802816L*x0))];
                            auto tmp2 = tmp1 / 4;
                            auto tmp3 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x1, 2L)));
                            auto tmp4 = c10::convert<int>(std::min(28L, 1L + (c10::div_floor_integer(x1, 2L))));
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                            auto tmp7 = c10::convert<int>(std::min(28L, 1L + (c10::div_floor_integer(x2, 2L))));
                            auto tmp8 = tmp6 < tmp7;
                            auto tmp9 = tmp5 & tmp8;
                            auto tmp10 = static_cast<float>(0.0);
                            auto tmp11 = tmp9 ? tmp2 : tmp10;
                            auto tmp12 = decltype(tmp0)(tmp0 + tmp11);
                            auto tmp13 = static_cast<float>(0.9805806756909201);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            in_out_ptr1[static_cast<long>(x3 + (256L*x2) + (14336L*x1) + (802816L*x0))] = tmp16;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (256L*x2) + (802816L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (256L*x2) + (802816L*x0)));
                            auto tmp1 = static_cast<float>(0.2);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp4 = static_cast<float>(2.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp8 = tmp6 * tmp7;
                            tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_71 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_div_mul_sigmoid_72 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (802816L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = static_cast<float>(0.2);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = decltype(tmp7)(1)/(decltype(tmp7)(1) + tmp7.neg().exp());
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = static_cast<float>(3136.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        auto tmp14 = tmp9 + tmp13;
                        tmp14.store(out_ptr0 + static_cast<long>(x1 + (256L*x2) + (802816L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.22351616621017456);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                auto tmp2 = in_ptr2[static_cast<long>(x0)];
                auto tmp5 = out_ptr1[static_cast<long>(x0)];
                auto tmp8 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = out_ptr0[static_cast<long>(x0)];
                auto tmp18 = in_ptr4[static_cast<long>(x0)];
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 - tmp3;
                auto tmp6 = static_cast<float>(0.015625);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp4 * tmp11;
                auto tmp13 = tmp0 - tmp12;
                auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp13 - tmp16;
                auto tmp19 = static_cast<float>(0.22351616621017456);
                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp17 * tmp22;
                tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp2;
                auto tmp6 = tmp1 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp2 * tmp7;
                auto tmp9 = tmp0 * tmp8;
                tmp9.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
            auto tmp3 = static_cast<float>(0.07450538873672485);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.001736111111111111);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = static_cast<float>(0.07450538873672485);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp8 * tmp19;
                    auto tmp21 = tmp15 * tmp20;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
            auto tmp3 = static_cast<float>(0.07450538873672485);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.001736111111111111);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = static_cast<float>(0.07450538873672485);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp8 * tmp19;
                    auto tmp21 = tmp15 * tmp20;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_mul_native_batch_norm_backward_view_76 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.1580497968320339);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = in_ptr2[static_cast<long>(x0)];
                auto tmp5 = out_ptr1[static_cast<long>(x0)];
                auto tmp8 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = out_ptr0[static_cast<long>(x0)];
                auto tmp18 = in_ptr4[static_cast<long>(x0)];
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 - tmp3;
                auto tmp6 = static_cast<float>(0.0078125);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp4 * tmp11;
                auto tmp13 = tmp0 - tmp12;
                auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp13 - tmp16;
                auto tmp19 = static_cast<float>(0.1580497968320339);
                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp17 * tmp22;
                tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 - tmp3;
                    auto tmp5 = tmp0 * tmp4;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.1580497968320339);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = in_ptr2[static_cast<long>(x0)];
                auto tmp5 = out_ptr1[static_cast<long>(x0)];
                auto tmp8 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = out_ptr0[static_cast<long>(x0)];
                auto tmp18 = in_ptr4[static_cast<long>(x0)];
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 - tmp3;
                auto tmp6 = static_cast<float>(0.0078125);
                auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp4 * tmp11;
                auto tmp13 = tmp0 - tmp12;
                auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp13 - tmp16;
                auto tmp19 = static_cast<float>(0.1580497968320339);
                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                auto tmp21 = decltype(tmp8)(tmp8 * tmp20);
                auto tmp22 = at::vec::Vectorized<float>(tmp21);
                auto tmp23 = tmp17 * tmp22;
                tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp7 = decltype(tmp6)(1)/(decltype(tmp6)(1) + tmp6.neg().exp());
                auto tmp8 = tmp4 - tmp7;
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = tmp9 + tmp4;
                auto tmp11 = tmp7 * tmp10;
                auto tmp12 = tmp5 * tmp11;
                tmp12.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((64L*(static_cast<long>(x1) % static_cast<long>(9L))) + (576L*x0) + (576L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.07450538873672485);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (64L*x1) + (576L*x0) + (576L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.001736111111111111);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = static_cast<float>(0.07450538873672485);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp8 * tmp19;
                    auto tmp21 = tmp15 * tmp20;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (576L*x0) + (576L*x0_inner))] = tmpbuf[x0_inner]; }
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x1) % static_cast<long>(9L))) + (288L*x0) + (288L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((32L*(static_cast<long>(x1) % static_cast<long>(9L))) + (288L*x0) + (288L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
            auto tmp3 = static_cast<float>(0.10536653122135592);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (32L*x1) + (288L*x0) + (288L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (32L*x1) + (288L*x0) + (288L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
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
                    auto tmp17 = static_cast<float>(0.10536653122135592);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp8 * tmp19;
                    auto tmp21 = tmp15 * tmp20;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (288L*x0) + (288L*x0_inner))] = tmpbuf[x0_inner]; }
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(9L))) + (144L*x0) + (144L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((16L*(static_cast<long>(x1) % static_cast<long>(9L))) + (144L*x0) + (144L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.1490107774734497);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (16L*x1) + (144L*x0) + (144L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (16L*x1) + (144L*x0) + (144L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.006944444444444444);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = static_cast<float>(0.1490107774734497);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp8 * tmp19;
                    auto tmp21 = tmp15 * tmp20;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (144L*x0) + (144L*x0_inner))] = tmpbuf[x0_inner]; }
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
}
''')


cpp_fused_mul_native_batch_norm_backward_view_81 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(27L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((3L*(static_cast<long>(x1) % static_cast<long>(9L))) + (27L*x0) + (27L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((3L*(static_cast<long>(x1) % static_cast<long>(9L))) + (27L*x0) + (27L*x0_inner) + (c10::div_floor_integer(x1, 9L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = static_cast<float>(0.34412564994580647);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 * tmp4;
            tmp5.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(3L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (3L*x1) + (27L*x0) + (27L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>(x2 + (3L*x1) + (27L*x0) + (27L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.037037037037037035);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = static_cast<float>(0.34412564994580647);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp8 * tmp19;
                    auto tmp21 = tmp15 * tmp20;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp21.store(tmpbuf); for (long x0_inner = 0; x0_inner < 8; x0_inner++) out_ptr3[static_cast<long>(x1 + (9L*x2) + (27L*x0) + (27L*x0_inner))] = tmpbuf[x0_inner]; }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_174, primals_176, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, primals_204, primals_206, primals_208, primals_210, primals_212, primals_214, primals_216, primals_218, primals_222, squeeze_1, view_2, convolution, mul_3, squeeze_3, view_5, convolution_1, mul_7, squeeze_5, view_8, convolution_2, mul_11, squeeze_7, view_11, convolution_3, mul_16, squeeze_9, view_14, squeeze_11, view_17, convolution_5, mul_23, squeeze_13, view_20, convolution_6, mul_27, squeeze_15, view_23, convolution_7, mul_31, squeeze_17, view_26, convolution_8, mean, relu, convolution_10, mul_39, avg_pool2d, squeeze_19, view_29, squeeze_21, view_32, convolution_12, mul_46, squeeze_23, view_35, convolution_13, mul_50, squeeze_25, view_38, convolution_14, mul_54, squeeze_27, view_41, convolution_15, mean_1, relu_1, convolution_17, mul_62, squeeze_29, view_44, convolution_18, mul_66, squeeze_31, view_47, convolution_19, mul_70, squeeze_33, view_50, convolution_20, mul_74, squeeze_35, view_53, convolution_21, mean_2, relu_2, convolution_23, mul_82, avg_pool2d_1, squeeze_37, view_56, squeeze_39, view_59, convolution_25, mul_89, squeeze_41, view_62, convolution_26, mul_93, squeeze_43, view_65, convolution_27, mul_97, squeeze_45, view_68, convolution_28, mean_3, relu_3, convolution_30, mul_105, squeeze_47, view_71, convolution_31, mul_109, squeeze_49, view_74, convolution_32, mul_113, squeeze_51, view_77, convolution_33, mul_117, squeeze_53, view_80, convolution_34, mean_4, relu_4, convolution_36, mul_125, squeeze_55, view_83, convolution_37, mul_129, squeeze_57, view_86, convolution_38, mul_133, squeeze_59, view_89, convolution_39, mul_137, squeeze_61, view_92, convolution_40, mean_5, relu_5, convolution_42, mul_145, squeeze_63, view_95, convolution_43, mul_149, squeeze_65, view_98, convolution_44, mul_153, squeeze_67, view_101, convolution_45, mul_157, squeeze_69, view_104, convolution_46, mean_6, relu_6, convolution_48, mul_165, squeeze_71, view_107, convolution_49, mul_169, squeeze_73, view_110, convolution_50, mul_173, squeeze_75, view_113, convolution_51, mul_177, squeeze_77, view_116, convolution_52, mean_7, relu_7, convolution_54, mul_185, squeeze_79, view_119, convolution_55, mul_189, squeeze_81, view_122, convolution_56, mul_193, squeeze_83, view_125, convolution_57, mul_197, squeeze_85, view_128, convolution_58, mean_8, relu_8, convolution_60, mul_205, avg_pool2d_2, squeeze_87, view_131, squeeze_89, view_134, convolution_62, mul_212, squeeze_91, view_137, convolution_63, mul_216, squeeze_93, view_140, convolution_64, mul_220, squeeze_95, view_143, convolution_65, mean_9, relu_9, convolution_67, mul_228, squeeze_97, view_146, convolution_68, mul_232, squeeze_99, view_149, convolution_69, mul_236, squeeze_101, view_152, convolution_70, mul_240, squeeze_103, view_155, convolution_71, mean_10, relu_10, convolution_73, mul_248, squeeze_105, view_158, convolution_74, mul_252, squeeze_107, view_161, convolution_75, mul_256, squeeze_109, view_164, convolution_76, mul_260, squeeze_111, view_167, convolution_77, mean_11, relu_11, convolution_79, add_67, squeeze_113, view_170, convolution_80, clone_28, permute_1, unsqueeze_58, unsqueeze_66, unsqueeze_74, unsqueeze_82, unsqueeze_90, mul_341, unsqueeze_98, unsqueeze_106, unsqueeze_114, unsqueeze_122, mul_400, unsqueeze_130, unsqueeze_138, unsqueeze_146, unsqueeze_154, unsqueeze_162, mul_469, unsqueeze_170, unsqueeze_178, unsqueeze_186, unsqueeze_194, mul_528, unsqueeze_202, unsqueeze_210, unsqueeze_218, unsqueeze_226, mul_587, unsqueeze_234, unsqueeze_242, unsqueeze_250, unsqueeze_258, mul_646, unsqueeze_266, unsqueeze_274, unsqueeze_282, unsqueeze_290, mul_705, unsqueeze_298, unsqueeze_306, unsqueeze_314, unsqueeze_322, mul_764, unsqueeze_330, unsqueeze_338, unsqueeze_346, unsqueeze_354, unsqueeze_362, mul_833, unsqueeze_370, unsqueeze_378, unsqueeze_386, unsqueeze_394, mul_892, unsqueeze_402, unsqueeze_410, unsqueeze_418, unsqueeze_426, unsqueeze_434, mul_961, unsqueeze_442, unsqueeze_450, unsqueeze_458, unsqueeze_466, unsqueeze_474, unsqueeze_482, unsqueeze_490, unsqueeze_498, unsqueeze_506, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_2, (16, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_4, (32, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_5, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_7, (64, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_8, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_10, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_11, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_13, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_14, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_16, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_17, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_19, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_20, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_22, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_23, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_25, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_26, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_28, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_29, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_31, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_32, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_34, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_35, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_37, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_38, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_40, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_41, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_43, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_44, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_46, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_47, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_49, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_50, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_52, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_53, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_55, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_56, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_58, (384, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_59, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_61, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_62, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_64, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_65, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_67, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_68, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_70, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_71, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_73, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_74, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_76, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_77, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_79, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_80, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_82, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_83, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_85, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_86, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_88, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_89, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_91, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_92, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_94, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_95, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_97, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_98, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_100, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_101, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_103, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_104, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_106, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_107, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_109, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_110, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_112, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_113, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_115, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_116, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_118, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_119, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_121, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_122, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_124, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_125, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_127, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_128, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_130, (1536, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_131, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_133, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_134, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_136, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_137, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_139, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_140, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_142, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_143, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_145, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_146, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_148, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_149, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_151, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_152, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_154, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_155, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_157, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_158, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_160, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_161, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_163, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_164, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_166, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_167, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_169, (2304, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_170, (2304, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_172, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_174, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_176, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_178, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_180, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_182, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_184, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_186, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_188, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_190, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_192, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_194, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_196, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_198, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_200, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_202, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_204, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_206, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_208, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_210, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_212, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_214, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_216, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_218, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_222, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(squeeze_1, (16, ), (1, ))
    assert_size_stride(view_2, (16, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(convolution, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(mul_3, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_3, (32, ), (1, ))
    assert_size_stride(view_5, (32, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(convolution_1, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(mul_7, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_5, (64, ), (1, ))
    assert_size_stride(view_8, (64, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(convolution_2, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(mul_11, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_7, (128, ), (1, ))
    assert_size_stride(view_11, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_3, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(mul_16, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_9, (256, ), (1, ))
    assert_size_stride(view_14, (256, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(squeeze_11, (64, ), (1, ))
    assert_size_stride(view_17, (64, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(convolution_5, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(mul_23, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_13, (64, ), (1, ))
    assert_size_stride(view_20, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_6, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(mul_27, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_15, (64, ), (1, ))
    assert_size_stride(view_23, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_7, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(mul_31, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_17, (256, ), (1, ))
    assert_size_stride(view_26, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(convolution_8, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(mean, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(relu, (8, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(convolution_10, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(mul_39, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(avg_pool2d, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_19, (512, ), (1, ))
    assert_size_stride(view_29, (512, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(squeeze_21, (128, ), (1, ))
    assert_size_stride(view_32, (128, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_12, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(mul_46, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_23, (128, ), (1, ))
    assert_size_stride(view_35, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_13, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(mul_50, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_25, (128, ), (1, ))
    assert_size_stride(view_38, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_14, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(mul_54, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_27, (512, ), (1, ))
    assert_size_stride(view_41, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(convolution_15, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(mean_1, (8, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(relu_1, (8, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(convolution_17, (8, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(mul_62, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_29, (128, ), (1, ))
    assert_size_stride(view_44, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convolution_18, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(mul_66, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_31, (128, ), (1, ))
    assert_size_stride(view_47, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_19, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(mul_70, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_33, (128, ), (1, ))
    assert_size_stride(view_50, (128, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_20, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(mul_74, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_35, (512, ), (1, ))
    assert_size_stride(view_53, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(convolution_21, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(mean_2, (8, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(relu_2, (8, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(convolution_23, (8, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(mul_82, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(avg_pool2d_1, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_37, (1536, ), (1, ))
    assert_size_stride(view_56, (1536, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(squeeze_39, (384, ), (1, ))
    assert_size_stride(view_59, (384, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convolution_25, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(mul_89, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(squeeze_41, (384, ), (1, ))
    assert_size_stride(view_62, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_26, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(mul_93, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_43, (384, ), (1, ))
    assert_size_stride(view_65, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_27, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(mul_97, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_45, (1536, ), (1, ))
    assert_size_stride(view_68, (1536, 384, 1, 1), (384, 1, 384, 384))
    assert_size_stride(convolution_28, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(mean_3, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(relu_3, (8, 384, 1, 1), (384, 1, 384, 384))
    assert_size_stride(convolution_30, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(mul_105, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(squeeze_47, (384, ), (1, ))
    assert_size_stride(view_71, (384, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(convolution_31, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(mul_109, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_49, (384, ), (1, ))
    assert_size_stride(view_74, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_32, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(mul_113, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_51, (384, ), (1, ))
    assert_size_stride(view_77, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_33, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(mul_117, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_53, (1536, ), (1, ))
    assert_size_stride(view_80, (1536, 384, 1, 1), (384, 1, 384, 384))
    assert_size_stride(convolution_34, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(mean_4, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(relu_4, (8, 384, 1, 1), (384, 1, 384, 384))
    assert_size_stride(convolution_36, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(mul_125, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(squeeze_55, (384, ), (1, ))
    assert_size_stride(view_83, (384, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(convolution_37, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(mul_129, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_57, (384, ), (1, ))
    assert_size_stride(view_86, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_38, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(mul_133, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_59, (384, ), (1, ))
    assert_size_stride(view_89, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_39, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(mul_137, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_61, (1536, ), (1, ))
    assert_size_stride(view_92, (1536, 384, 1, 1), (384, 1, 384, 384))
    assert_size_stride(convolution_40, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(mean_5, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(relu_5, (8, 384, 1, 1), (384, 1, 384, 384))
    assert_size_stride(convolution_42, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(mul_145, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(squeeze_63, (384, ), (1, ))
    assert_size_stride(view_95, (384, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(convolution_43, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(mul_149, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_65, (384, ), (1, ))
    assert_size_stride(view_98, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_44, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(mul_153, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_67, (384, ), (1, ))
    assert_size_stride(view_101, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_45, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(mul_157, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_69, (1536, ), (1, ))
    assert_size_stride(view_104, (1536, 384, 1, 1), (384, 1, 384, 384))
    assert_size_stride(convolution_46, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(mean_6, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(relu_6, (8, 384, 1, 1), (384, 1, 384, 384))
    assert_size_stride(convolution_48, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(mul_165, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(squeeze_71, (384, ), (1, ))
    assert_size_stride(view_107, (384, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(convolution_49, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(mul_169, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_73, (384, ), (1, ))
    assert_size_stride(view_110, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_50, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(mul_173, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_75, (384, ), (1, ))
    assert_size_stride(view_113, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_51, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(mul_177, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_77, (1536, ), (1, ))
    assert_size_stride(view_116, (1536, 384, 1, 1), (384, 1, 384, 384))
    assert_size_stride(convolution_52, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(mean_7, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(relu_7, (8, 384, 1, 1), (384, 1, 384, 384))
    assert_size_stride(convolution_54, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(mul_185, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(squeeze_79, (384, ), (1, ))
    assert_size_stride(view_119, (384, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(convolution_55, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(mul_189, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_81, (384, ), (1, ))
    assert_size_stride(view_122, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_56, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(mul_193, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_83, (384, ), (1, ))
    assert_size_stride(view_125, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_57, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(mul_197, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_85, (1536, ), (1, ))
    assert_size_stride(view_128, (1536, 384, 1, 1), (384, 1, 384, 384))
    assert_size_stride(convolution_58, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(mean_8, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(relu_8, (8, 384, 1, 1), (384, 1, 384, 384))
    assert_size_stride(convolution_60, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(mul_205, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(avg_pool2d_2, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
    assert_size_stride(squeeze_87, (1536, ), (1, ))
    assert_size_stride(view_131, (1536, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(squeeze_89, (384, ), (1, ))
    assert_size_stride(view_134, (384, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(convolution_62, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(mul_212, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_91, (384, ), (1, ))
    assert_size_stride(view_137, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_63, (8, 384, 7, 7), (18816, 1, 2688, 384))
    assert_size_stride(mul_216, (8, 384, 7, 7), (18816, 1, 2688, 384))
    assert_size_stride(squeeze_93, (384, ), (1, ))
    assert_size_stride(view_140, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_64, (8, 384, 7, 7), (18816, 1, 2688, 384))
    assert_size_stride(mul_220, (8, 384, 7, 7), (18816, 1, 2688, 384))
    assert_size_stride(squeeze_95, (1536, ), (1, ))
    assert_size_stride(view_143, (1536, 384, 1, 1), (384, 1, 384, 384))
    assert_size_stride(convolution_65, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
    assert_size_stride(mean_9, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(relu_9, (8, 384, 1, 1), (384, 1, 384, 384))
    assert_size_stride(convolution_67, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(mul_228, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
    assert_size_stride(squeeze_97, (384, ), (1, ))
    assert_size_stride(view_146, (384, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(convolution_68, (8, 384, 7, 7), (18816, 1, 2688, 384))
    assert_size_stride(mul_232, (8, 384, 7, 7), (18816, 1, 2688, 384))
    assert_size_stride(squeeze_99, (384, ), (1, ))
    assert_size_stride(view_149, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_69, (8, 384, 7, 7), (18816, 1, 2688, 384))
    assert_size_stride(mul_236, (8, 384, 7, 7), (18816, 1, 2688, 384))
    assert_size_stride(squeeze_101, (384, ), (1, ))
    assert_size_stride(view_152, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_70, (8, 384, 7, 7), (18816, 1, 2688, 384))
    assert_size_stride(mul_240, (8, 384, 7, 7), (18816, 1, 2688, 384))
    assert_size_stride(squeeze_103, (1536, ), (1, ))
    assert_size_stride(view_155, (1536, 384, 1, 1), (384, 1, 384, 384))
    assert_size_stride(convolution_71, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
    assert_size_stride(mean_10, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(relu_10, (8, 384, 1, 1), (384, 1, 384, 384))
    assert_size_stride(convolution_73, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(mul_248, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
    assert_size_stride(squeeze_105, (384, ), (1, ))
    assert_size_stride(view_158, (384, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(convolution_74, (8, 384, 7, 7), (18816, 1, 2688, 384))
    assert_size_stride(mul_252, (8, 384, 7, 7), (18816, 1, 2688, 384))
    assert_size_stride(squeeze_107, (384, ), (1, ))
    assert_size_stride(view_161, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_75, (8, 384, 7, 7), (18816, 1, 2688, 384))
    assert_size_stride(mul_256, (8, 384, 7, 7), (18816, 1, 2688, 384))
    assert_size_stride(squeeze_109, (384, ), (1, ))
    assert_size_stride(view_164, (384, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_76, (8, 384, 7, 7), (18816, 1, 2688, 384))
    assert_size_stride(mul_260, (8, 384, 7, 7), (18816, 1, 2688, 384))
    assert_size_stride(squeeze_111, (1536, ), (1, ))
    assert_size_stride(view_167, (1536, 384, 1, 1), (384, 1, 384, 384))
    assert_size_stride(convolution_77, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
    assert_size_stride(mean_11, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(relu_11, (8, 384, 1, 1), (384, 1, 384, 384))
    assert_size_stride(convolution_79, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(add_67, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
    assert_size_stride(squeeze_113, (2304, ), (1, ))
    assert_size_stride(view_170, (2304, 1536, 1, 1), (1536, 1, 1536, 1536))
    assert_size_stride(convolution_80, (8, 2304, 7, 7), (112896, 1, 16128, 2304))
    assert_size_stride(clone_28, (8, 2304), (2304, 1))
    assert_size_stride(permute_1, (1000, 2304), (2304, 1))
    assert_size_stride(unsqueeze_58, (1, 2304, 1), (2304, 1, 1))
    assert_size_stride(unsqueeze_66, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_74, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_82, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_90, (1, 384, 1), (384, 1, 1))
    assert_size_stride(mul_341, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
    assert_size_stride(unsqueeze_98, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_106, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_114, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_122, (1, 384, 1), (384, 1, 1))
    assert_size_stride(mul_400, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
    assert_size_stride(unsqueeze_130, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_138, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_146, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_154, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_162, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(mul_469, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(unsqueeze_170, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_178, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_186, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_194, (1, 384, 1), (384, 1, 1))
    assert_size_stride(mul_528, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(unsqueeze_202, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_210, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_218, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_226, (1, 384, 1), (384, 1, 1))
    assert_size_stride(mul_587, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(unsqueeze_234, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_242, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_250, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 384, 1), (384, 1, 1))
    assert_size_stride(mul_646, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(unsqueeze_266, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_274, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_282, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_290, (1, 384, 1), (384, 1, 1))
    assert_size_stride(mul_705, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(unsqueeze_298, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_306, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_314, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_322, (1, 384, 1), (384, 1, 1))
    assert_size_stride(mul_764, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(unsqueeze_330, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(unsqueeze_338, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_346, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_354, (1, 384, 1), (384, 1, 1))
    assert_size_stride(unsqueeze_362, (1, 1536, 1), (1536, 1, 1))
    assert_size_stride(mul_833, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(unsqueeze_370, (1, 512, 1), (512, 1, 1))
    assert_size_stride(unsqueeze_378, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_386, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_394, (1, 128, 1), (128, 1, 1))
    assert_size_stride(mul_892, (8, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(unsqueeze_402, (1, 512, 1), (512, 1, 1))
    assert_size_stride(unsqueeze_410, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_418, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_426, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_434, (1, 512, 1), (512, 1, 1))
    assert_size_stride(mul_961, (8, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(unsqueeze_442, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_450, (1, 64, 1), (64, 1, 1))
    assert_size_stride(unsqueeze_458, (1, 64, 1), (64, 1, 1))
    assert_size_stride(unsqueeze_466, (1, 64, 1), (64, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 256, 1), (256, 1, 1))
    assert_size_stride(unsqueeze_482, (1, 128, 1), (128, 1, 1))
    assert_size_stride(unsqueeze_490, (1, 64, 1), (64, 1, 1))
    assert_size_stride(unsqueeze_498, (1, 32, 1), (32, 1, 1))
    assert_size_stride(unsqueeze_506, (1, 16, 1), (16, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf0)
    del permute_1
    buf1 = empty((1000, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_28, out=buf1)
    del clone_28
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty((8, 2304, 7, 7), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_div_fill_mul_sigmoid_sub_sum_0(c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_80.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()))
    del convolution_80
    del tangents_1
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf4 = aten.convolution_backward(buf3, add_67, view_170, [2304], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_67
    del buf3
    del view_170
    buf5 = buf4[0]
    buf6 = buf4[1]
    buf7 = buf4[2]
    del buf4
    buf8 = empty((2304, ), device='cpu', dtype=torch.float32)
    buf9 = empty((2304, ), device='cpu', dtype=torch.float32)
    buf10 = empty((2304, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf11 = reinterpret_tensor(buf6, (2304, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf6  # reuse
    buf12 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cpu', dtype=torch.float32)
    buf13 = reinterpret_tensor(buf12, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf12  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_1(c_void_p(buf11.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(unsqueeze_58.data_ptr()), c_void_p(squeeze_113.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(convolution_77.data_ptr()), c_void_p(convolution_79.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()))
    del buf8
    del buf9
    del convolution_77
    del primals_169
    del primals_170
    del squeeze_113
    del unsqueeze_58
    # Source Nodes: [sigmoid_11], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf14 = aten.convolution_backward(buf13, relu_11, primals_218, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf13
    del primals_218
    buf15 = buf14[0]
    buf16 = buf14[1]
    buf17 = buf14[2]
    del buf14
    buf18 = buf15; del buf15  # reuse
    cpp_fused_convolution_backward_threshold_backward_2(c_void_p(buf18.data_ptr()), c_void_p(relu_11.data_ptr()))
    del relu_11
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf19 = aten.convolution_backward(buf18, mean_11, primals_216, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf18
    del mean_11
    del primals_216
    buf20 = buf19[0]
    buf21 = buf19[1]
    buf22 = buf19[2]
    del buf19
    buf23 = empty_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_div_mul_sigmoid_3(c_void_p(buf5.data_ptr()), c_void_p(convolution_79.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf23.data_ptr()))
    del convolution_79
    # Source Nodes: [sigmoid_11], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf24 = aten.convolution_backward(buf23, mul_260, view_167, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mul_260
    del view_167
    buf25 = buf24[0]
    buf26 = buf24[1]
    buf27 = buf24[2]
    del buf24
    buf28 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf29 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf30 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf31 = reinterpret_tensor(buf26, (1536, 384, 1, 1), (384, 1, 1, 1), 0); del buf26  # reuse
    buf32 = buf25; del buf25  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_4(c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(unsqueeze_66.data_ptr()), c_void_p(squeeze_111.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(convolution_76.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()))
    del convolution_76
    del primals_166
    del primals_167
    del squeeze_111
    del unsqueeze_66
    # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act3], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.silu, aten.sub]
    buf33 = aten.convolution_backward(buf32, mul_256, view_164, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf32
    del mul_256
    del view_164
    buf34 = buf33[0]
    buf35 = buf33[1]
    buf36 = buf33[2]
    del buf33
    buf37 = empty((384, ), device='cpu', dtype=torch.float32)
    buf38 = empty((384, ), device='cpu', dtype=torch.float32)
    buf39 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf40 = empty((384, 64, 3, 3), device='cpu', dtype=torch.float32)
    buf41 = buf34; del buf34  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_5(c_void_p(buf41.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(unsqueeze_74.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(convolution_75.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()))
    del convolution_75
    del primals_163
    del primals_164
    del squeeze_109
    del unsqueeze_74
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf42 = aten.convolution_backward(buf41, mul_252, view_161, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf41
    del mul_252
    del view_161
    buf43 = buf42[0]
    buf44 = buf42[1]
    buf45 = buf42[2]
    del buf42
    buf46 = buf38; del buf38  # reuse
    buf47 = buf37; del buf37  # reuse
    buf48 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf49 = reinterpret_tensor(buf35, (384, 64, 3, 3), (576, 9, 3, 1), 0); del buf35  # reuse
    buf50 = buf43; del buf43  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_6(c_void_p(buf50.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(unsqueeze_82.data_ptr()), c_void_p(squeeze_107.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(convolution_74.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()))
    del convolution_74
    del primals_160
    del primals_161
    del squeeze_107
    del unsqueeze_82
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf51 = aten.convolution_backward(buf50, mul_248, view_158, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf50
    del mul_248
    del view_158
    buf52 = buf51[0]
    buf53 = buf51[1]
    buf54 = buf51[2]
    del buf51
    buf55 = buf47; del buf47  # reuse
    buf56 = buf46; del buf46  # reuse
    buf57 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf58 = reinterpret_tensor(buf53, (384, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf53  # reuse
    buf59 = reinterpret_tensor(buf20, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf20  # reuse
    buf60 = reinterpret_tensor(buf59, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf59  # reuse
    cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_7(c_void_p(buf58.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(unsqueeze_90.data_ptr()), c_void_p(squeeze_105.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(mul_341.data_ptr()), c_void_p(convolution_71.data_ptr()), c_void_p(convolution_73.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()))
    del convolution_71
    del primals_157
    del primals_158
    del squeeze_105
    del unsqueeze_90
    # Source Nodes: [sigmoid_10], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf61 = aten.convolution_backward(buf60, relu_10, primals_214, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf60
    del primals_214
    buf62 = buf61[0]
    buf63 = buf61[1]
    buf64 = buf61[2]
    del buf61
    buf65 = buf62; del buf62  # reuse
    cpp_fused_convolution_backward_threshold_backward_8(c_void_p(buf65.data_ptr()), c_void_p(relu_10.data_ptr()))
    del relu_10
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf66 = aten.convolution_backward(buf65, mean_10, primals_212, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf65
    del mean_10
    del primals_212
    buf67 = buf66[0]
    buf68 = buf66[1]
    buf69 = buf66[2]
    del buf66
    buf70 = buf23; del buf23  # reuse
    cpp_fused_add_convolution_backward_div_mul_sigmoid_9(c_void_p(buf5.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(mul_341.data_ptr()), c_void_p(convolution_73.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf70.data_ptr()))
    del convolution_73
    # Source Nodes: [sigmoid_10], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf71 = aten.convolution_backward(buf70, mul_240, view_155, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf70
    del mul_240
    del view_155
    buf72 = buf71[0]
    buf73 = buf71[1]
    buf74 = buf71[2]
    del buf71
    buf75 = buf29; del buf29  # reuse
    buf76 = buf28; del buf28  # reuse
    buf77 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf78 = reinterpret_tensor(buf73, (1536, 384, 1, 1), (384, 1, 1, 1), 0); del buf73  # reuse
    buf79 = buf72; del buf72  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_10(c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(unsqueeze_98.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(convolution_70.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()))
    del convolution_70
    del primals_154
    del primals_155
    del squeeze_103
    del unsqueeze_98
    # Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act3], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.silu, aten.sub]
    buf80 = aten.convolution_backward(buf79, mul_236, view_152, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf79
    del mul_236
    del view_152
    buf81 = buf80[0]
    buf82 = buf80[1]
    buf83 = buf80[2]
    del buf80
    buf84 = buf56; del buf56  # reuse
    buf85 = buf55; del buf55  # reuse
    buf86 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf87 = reinterpret_tensor(buf44, (384, 64, 3, 3), (576, 9, 3, 1), 0); del buf44  # reuse
    buf88 = buf81; del buf81  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_11(c_void_p(buf88.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(unsqueeze_106.data_ptr()), c_void_p(squeeze_101.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(convolution_69.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()))
    del convolution_69
    del primals_151
    del primals_152
    del squeeze_101
    del unsqueeze_106
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf89 = aten.convolution_backward(buf88, mul_232, view_149, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf88
    del mul_232
    del view_149
    buf90 = buf89[0]
    buf91 = buf89[1]
    buf92 = buf89[2]
    del buf89
    buf93 = buf85; del buf85  # reuse
    buf94 = buf84; del buf84  # reuse
    buf95 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf96 = reinterpret_tensor(buf82, (384, 64, 3, 3), (576, 9, 3, 1), 0); del buf82  # reuse
    buf97 = buf90; del buf90  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_12(c_void_p(buf97.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(unsqueeze_114.data_ptr()), c_void_p(squeeze_99.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(convolution_68.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()))
    del convolution_68
    del primals_148
    del primals_149
    del squeeze_99
    del unsqueeze_114
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf98 = aten.convolution_backward(buf97, mul_228, view_146, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf97
    del mul_228
    del view_146
    buf99 = buf98[0]
    buf100 = buf98[1]
    buf101 = buf98[2]
    del buf98
    buf102 = buf94; del buf94  # reuse
    buf103 = buf93; del buf93  # reuse
    buf104 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf105 = reinterpret_tensor(buf100, (384, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf100  # reuse
    buf106 = buf5; del buf5  # reuse
    buf107 = reinterpret_tensor(buf67, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf67  # reuse
    buf108 = reinterpret_tensor(buf107, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf107  # reuse
    cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_13(c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(unsqueeze_122.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(mul_341.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(mul_400.data_ptr()), c_void_p(convolution_65.data_ptr()), c_void_p(convolution_67.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()))
    del buf52
    del convolution_65
    del mul_341
    del mul_400
    del primals_145
    del primals_146
    del squeeze_97
    del unsqueeze_122
    # Source Nodes: [sigmoid_9], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf109 = aten.convolution_backward(buf108, relu_9, primals_210, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf108
    del primals_210
    buf110 = buf109[0]
    buf111 = buf109[1]
    buf112 = buf109[2]
    del buf109
    buf113 = buf110; del buf110  # reuse
    cpp_fused_convolution_backward_threshold_backward_14(c_void_p(buf113.data_ptr()), c_void_p(relu_9.data_ptr()))
    del relu_9
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf114 = aten.convolution_backward(buf113, mean_9, primals_208, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf113
    del mean_9
    del primals_208
    buf115 = buf114[0]
    buf116 = buf114[1]
    buf117 = buf114[2]
    del buf114
    buf118 = buf99; del buf99  # reuse
    cpp_fused_add_convolution_backward_div_mul_sigmoid_15(c_void_p(buf106.data_ptr()), c_void_p(convolution_67.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf118.data_ptr()))
    del convolution_67
    # Source Nodes: [sigmoid_9], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf119 = aten.convolution_backward(buf118, mul_220, view_143, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf118
    del mul_220
    del view_143
    buf120 = buf119[0]
    buf121 = buf119[1]
    buf122 = buf119[2]
    del buf119
    buf123 = buf76; del buf76  # reuse
    buf124 = buf75; del buf75  # reuse
    buf125 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf126 = reinterpret_tensor(buf121, (1536, 384, 1, 1), (384, 1, 1, 1), 0); del buf121  # reuse
    buf127 = buf120; del buf120  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_16(c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(unsqueeze_130.data_ptr()), c_void_p(squeeze_95.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(convolution_64.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()))
    del convolution_64
    del primals_142
    del primals_143
    del squeeze_95
    del unsqueeze_130
    # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act3], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.silu, aten.sub]
    buf128 = aten.convolution_backward(buf127, mul_216, view_140, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf127
    del mul_216
    del view_140
    buf129 = buf128[0]
    buf130 = buf128[1]
    buf131 = buf128[2]
    del buf128
    buf132 = buf103; del buf103  # reuse
    buf133 = buf102; del buf102  # reuse
    buf134 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf135 = reinterpret_tensor(buf91, (384, 64, 3, 3), (576, 9, 3, 1), 0); del buf91  # reuse
    buf136 = buf129; del buf129  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_17(c_void_p(buf136.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(unsqueeze_138.data_ptr()), c_void_p(squeeze_93.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(convolution_63.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()))
    del convolution_63
    del primals_139
    del primals_140
    del squeeze_93
    del unsqueeze_138
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf137 = aten.convolution_backward(buf136, mul_212, view_137, [384], [2, 2], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf136
    del mul_212
    del view_137
    buf138 = buf137[0]
    buf139 = buf137[1]
    buf140 = buf137[2]
    del buf137
    buf141 = buf133; del buf133  # reuse
    buf142 = buf132; del buf132  # reuse
    buf143 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf144 = reinterpret_tensor(buf130, (384, 64, 3, 3), (576, 9, 3, 1), 0); del buf130  # reuse
    buf145 = buf138; del buf138  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_18(c_void_p(buf145.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(unsqueeze_146.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(convolution_62.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()))
    del convolution_62
    del primals_136
    del primals_137
    del squeeze_91
    del unsqueeze_146
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf146 = aten.convolution_backward(buf145, mul_205, view_134, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf145
    del mul_205
    del view_134
    buf147 = buf146[0]
    buf148 = buf146[1]
    buf149 = buf146[2]
    del buf146
    buf150 = buf142; del buf142  # reuse
    buf151 = buf141; del buf141  # reuse
    buf152 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf153 = reinterpret_tensor(buf148, (384, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf148  # reuse
    cpp_fused_mul_native_batch_norm_backward_view_19(c_void_p(buf153.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(unsqueeze_154.data_ptr()), c_void_p(squeeze_89.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()))
    del primals_133
    del primals_134
    del squeeze_89
    del unsqueeze_154
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf154 = aten.convolution_backward(buf106, avg_pool2d_2, view_131, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del avg_pool2d_2
    del buf106
    del view_131
    buf155 = buf154[0]
    buf156 = buf154[1]
    buf157 = buf154[2]
    del buf154
    buf158 = buf124; del buf124  # reuse
    buf159 = buf123; del buf123  # reuse
    buf160 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf161 = reinterpret_tensor(buf156, (1536, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf156  # reuse
    buf162 = reinterpret_tensor(buf115, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf115  # reuse
    buf163 = reinterpret_tensor(buf162, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf162  # reuse
    cpp_fused_add_avg_pool2d_backward_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_20(c_void_p(buf161.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(unsqueeze_162.data_ptr()), c_void_p(squeeze_87.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(mul_469.data_ptr()), c_void_p(convolution_58.data_ptr()), c_void_p(convolution_60.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()))
    del convolution_58
    del primals_130
    del primals_131
    del squeeze_87
    del unsqueeze_162
    # Source Nodes: [sigmoid_8], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf164 = aten.convolution_backward(buf163, relu_8, primals_206, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf163
    del primals_206
    buf165 = buf164[0]
    buf166 = buf164[1]
    buf167 = buf164[2]
    del buf164
    buf168 = buf165; del buf165  # reuse
    cpp_fused_convolution_backward_threshold_backward_21(c_void_p(buf168.data_ptr()), c_void_p(relu_8.data_ptr()))
    del relu_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf169 = aten.convolution_backward(buf168, mean_8, primals_204, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf168
    del mean_8
    del primals_204
    buf170 = buf169[0]
    buf171 = buf169[1]
    buf172 = buf169[2]
    del buf169
    buf173 = empty_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_add_avg_pool2d_backward_convolution_backward_div_mul_sigmoid_22(c_void_p(buf147.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(mul_469.data_ptr()), c_void_p(convolution_60.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf173.data_ptr()))
    del convolution_60
    # Source Nodes: [sigmoid_8], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf174 = aten.convolution_backward(buf173, mul_197, view_128, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf173
    del mul_197
    del view_128
    buf175 = buf174[0]
    buf176 = buf174[1]
    buf177 = buf174[2]
    del buf174
    buf178 = buf159; del buf159  # reuse
    buf179 = buf158; del buf158  # reuse
    buf180 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf181 = reinterpret_tensor(buf176, (1536, 384, 1, 1), (384, 1, 1, 1), 0); del buf176  # reuse
    buf182 = buf175; del buf175  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_23(c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(unsqueeze_170.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(convolution_57.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()))
    del convolution_57
    del primals_127
    del primals_128
    del squeeze_85
    del unsqueeze_170
    # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act3], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.silu, aten.sub]
    buf183 = aten.convolution_backward(buf182, mul_193, view_125, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf182
    del mul_193
    del view_125
    buf184 = buf183[0]
    buf185 = buf183[1]
    buf186 = buf183[2]
    del buf183
    buf187 = buf151; del buf151  # reuse
    buf188 = buf150; del buf150  # reuse
    buf189 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf190 = reinterpret_tensor(buf139, (384, 64, 3, 3), (576, 9, 3, 1), 0); del buf139  # reuse
    buf191 = buf184; del buf184  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_24(c_void_p(buf191.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(unsqueeze_178.data_ptr()), c_void_p(squeeze_83.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(convolution_56.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()))
    del convolution_56
    del primals_124
    del primals_125
    del squeeze_83
    del unsqueeze_178
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf192 = aten.convolution_backward(buf191, mul_189, view_122, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf191
    del mul_189
    del view_122
    buf193 = buf192[0]
    buf194 = buf192[1]
    buf195 = buf192[2]
    del buf192
    buf196 = buf188; del buf188  # reuse
    buf197 = buf187; del buf187  # reuse
    buf198 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf199 = reinterpret_tensor(buf185, (384, 64, 3, 3), (576, 9, 3, 1), 0); del buf185  # reuse
    buf200 = buf193; del buf193  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_25(c_void_p(buf200.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(unsqueeze_186.data_ptr()), c_void_p(squeeze_81.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(convolution_55.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()))
    del convolution_55
    del primals_121
    del primals_122
    del squeeze_81
    del unsqueeze_186
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf201 = aten.convolution_backward(buf200, mul_185, view_119, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf200
    del mul_185
    del view_119
    buf202 = buf201[0]
    buf203 = buf201[1]
    buf204 = buf201[2]
    del buf201
    buf205 = buf197; del buf197  # reuse
    buf206 = buf196; del buf196  # reuse
    buf207 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf208 = reinterpret_tensor(buf203, (384, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf203  # reuse
    buf209 = buf147; del buf147  # reuse
    buf210 = reinterpret_tensor(buf170, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf170  # reuse
    buf211 = reinterpret_tensor(buf210, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf210  # reuse
    cpp_fused_add_avg_pool2d_backward_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_26(c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(unsqueeze_194.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(mul_469.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(mul_528.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()))
    del buf155
    del convolution_52
    del mul_469
    del mul_528
    del primals_118
    del primals_119
    del squeeze_79
    del unsqueeze_194
    # Source Nodes: [sigmoid_7], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf212 = aten.convolution_backward(buf211, relu_7, primals_202, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf211
    del primals_202
    buf213 = buf212[0]
    buf214 = buf212[1]
    buf215 = buf212[2]
    del buf212
    buf216 = buf213; del buf213  # reuse
    cpp_fused_convolution_backward_threshold_backward_27(c_void_p(buf216.data_ptr()), c_void_p(relu_7.data_ptr()))
    del relu_7
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf217 = aten.convolution_backward(buf216, mean_7, primals_200, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf216
    del mean_7
    del primals_200
    buf218 = buf217[0]
    buf219 = buf217[1]
    buf220 = buf217[2]
    del buf217
    buf221 = buf202; del buf202  # reuse
    cpp_fused_add_convolution_backward_div_mul_sigmoid_28(c_void_p(buf209.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf221.data_ptr()))
    del convolution_54
    # Source Nodes: [sigmoid_7], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf222 = aten.convolution_backward(buf221, mul_177, view_116, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mul_177
    del view_116
    buf223 = buf222[0]
    buf224 = buf222[1]
    buf225 = buf222[2]
    del buf222
    buf226 = buf179; del buf179  # reuse
    buf227 = buf178; del buf178  # reuse
    buf228 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf229 = reinterpret_tensor(buf224, (1536, 384, 1, 1), (384, 1, 1, 1), 0); del buf224  # reuse
    buf230 = buf223; del buf223  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_29(c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(unsqueeze_202.data_ptr()), c_void_p(squeeze_77.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()))
    del convolution_51
    del primals_115
    del primals_116
    del squeeze_77
    del unsqueeze_202
    # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act3], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.silu, aten.sub]
    buf231 = aten.convolution_backward(buf230, mul_173, view_113, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf230
    del mul_173
    del view_113
    buf232 = buf231[0]
    buf233 = buf231[1]
    buf234 = buf231[2]
    del buf231
    buf235 = buf206; del buf206  # reuse
    buf236 = buf205; del buf205  # reuse
    buf237 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf238 = reinterpret_tensor(buf194, (384, 64, 3, 3), (576, 9, 3, 1), 0); del buf194  # reuse
    buf239 = buf232; del buf232  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_30(c_void_p(buf239.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(unsqueeze_210.data_ptr()), c_void_p(squeeze_75.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()))
    del convolution_50
    del primals_112
    del primals_113
    del squeeze_75
    del unsqueeze_210
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf240 = aten.convolution_backward(buf239, mul_169, view_110, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf239
    del mul_169
    del view_110
    buf241 = buf240[0]
    buf242 = buf240[1]
    buf243 = buf240[2]
    del buf240
    buf244 = buf236; del buf236  # reuse
    buf245 = buf235; del buf235  # reuse
    buf246 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf247 = reinterpret_tensor(buf233, (384, 64, 3, 3), (576, 9, 3, 1), 0); del buf233  # reuse
    buf248 = buf241; del buf241  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_31(c_void_p(buf248.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(unsqueeze_218.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()))
    del convolution_49
    del primals_109
    del primals_110
    del squeeze_73
    del unsqueeze_218
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf249 = aten.convolution_backward(buf248, mul_165, view_107, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf248
    del mul_165
    del view_107
    buf250 = buf249[0]
    buf251 = buf249[1]
    buf252 = buf249[2]
    del buf249
    buf253 = buf245; del buf245  # reuse
    buf254 = buf244; del buf244  # reuse
    buf255 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf256 = reinterpret_tensor(buf251, (384, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf251  # reuse
    buf257 = reinterpret_tensor(buf218, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf218  # reuse
    buf258 = reinterpret_tensor(buf257, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf257  # reuse
    cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_32(c_void_p(buf256.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(unsqueeze_226.data_ptr()), c_void_p(squeeze_71.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(mul_587.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()))
    del convolution_46
    del primals_106
    del primals_107
    del squeeze_71
    del unsqueeze_226
    # Source Nodes: [sigmoid_6], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf259 = aten.convolution_backward(buf258, relu_6, primals_198, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf258
    del primals_198
    buf260 = buf259[0]
    buf261 = buf259[1]
    buf262 = buf259[2]
    del buf259
    buf263 = buf260; del buf260  # reuse
    cpp_fused_convolution_backward_threshold_backward_33(c_void_p(buf263.data_ptr()), c_void_p(relu_6.data_ptr()))
    del relu_6
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf264 = aten.convolution_backward(buf263, mean_6, primals_196, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf263
    del mean_6
    del primals_196
    buf265 = buf264[0]
    buf266 = buf264[1]
    buf267 = buf264[2]
    del buf264
    buf268 = buf221; del buf221  # reuse
    cpp_fused_add_convolution_backward_div_mul_sigmoid_34(c_void_p(buf209.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(mul_587.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf268.data_ptr()))
    del convolution_48
    # Source Nodes: [sigmoid_6], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf269 = aten.convolution_backward(buf268, mul_157, view_104, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf268
    del mul_157
    del view_104
    buf270 = buf269[0]
    buf271 = buf269[1]
    buf272 = buf269[2]
    del buf269
    buf273 = buf227; del buf227  # reuse
    buf274 = buf226; del buf226  # reuse
    buf275 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf276 = reinterpret_tensor(buf271, (1536, 384, 1, 1), (384, 1, 1, 1), 0); del buf271  # reuse
    buf277 = buf270; del buf270  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_35(c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(unsqueeze_234.data_ptr()), c_void_p(squeeze_69.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()))
    del convolution_45
    del primals_103
    del primals_104
    del squeeze_69
    del unsqueeze_234
    # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act3], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.silu, aten.sub]
    buf278 = aten.convolution_backward(buf277, mul_153, view_101, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf277
    del mul_153
    del view_101
    buf279 = buf278[0]
    buf280 = buf278[1]
    buf281 = buf278[2]
    del buf278
    buf282 = buf254; del buf254  # reuse
    buf283 = buf253; del buf253  # reuse
    buf284 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf285 = reinterpret_tensor(buf242, (384, 64, 3, 3), (576, 9, 3, 1), 0); del buf242  # reuse
    buf286 = buf279; del buf279  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_36(c_void_p(buf286.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(unsqueeze_242.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()))
    del convolution_44
    del primals_100
    del primals_101
    del squeeze_67
    del unsqueeze_242
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf287 = aten.convolution_backward(buf286, mul_149, view_98, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf286
    del mul_149
    del view_98
    buf288 = buf287[0]
    buf289 = buf287[1]
    buf290 = buf287[2]
    del buf287
    buf291 = buf283; del buf283  # reuse
    buf292 = buf282; del buf282  # reuse
    buf293 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf294 = reinterpret_tensor(buf280, (384, 64, 3, 3), (576, 9, 3, 1), 0); del buf280  # reuse
    buf295 = buf288; del buf288  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_37(c_void_p(buf295.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(unsqueeze_250.data_ptr()), c_void_p(squeeze_65.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()))
    del convolution_43
    del primals_97
    del primals_98
    del squeeze_65
    del unsqueeze_250
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf296 = aten.convolution_backward(buf295, mul_145, view_95, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf295
    del mul_145
    del view_95
    buf297 = buf296[0]
    buf298 = buf296[1]
    buf299 = buf296[2]
    del buf296
    buf300 = buf292; del buf292  # reuse
    buf301 = buf291; del buf291  # reuse
    buf302 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf303 = reinterpret_tensor(buf298, (384, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf298  # reuse
    buf304 = buf209; del buf209  # reuse
    buf305 = reinterpret_tensor(buf265, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf265  # reuse
    buf306 = reinterpret_tensor(buf305, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf305  # reuse
    cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_38(c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(unsqueeze_258.data_ptr()), c_void_p(squeeze_63.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(mul_587.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(mul_646.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()))
    del buf250
    del convolution_40
    del mul_587
    del mul_646
    del primals_94
    del primals_95
    del squeeze_63
    del unsqueeze_258
    # Source Nodes: [sigmoid_5], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf307 = aten.convolution_backward(buf306, relu_5, primals_194, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf306
    del primals_194
    buf308 = buf307[0]
    buf309 = buf307[1]
    buf310 = buf307[2]
    del buf307
    buf311 = buf308; del buf308  # reuse
    cpp_fused_convolution_backward_threshold_backward_39(c_void_p(buf311.data_ptr()), c_void_p(relu_5.data_ptr()))
    del relu_5
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf312 = aten.convolution_backward(buf311, mean_5, primals_192, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf311
    del mean_5
    del primals_192
    buf313 = buf312[0]
    buf314 = buf312[1]
    buf315 = buf312[2]
    del buf312
    buf316 = buf297; del buf297  # reuse
    cpp_fused_add_convolution_backward_div_mul_sigmoid_40(c_void_p(buf304.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf316.data_ptr()))
    del convolution_42
    # Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf317 = aten.convolution_backward(buf316, mul_137, view_92, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mul_137
    del view_92
    buf318 = buf317[0]
    buf319 = buf317[1]
    buf320 = buf317[2]
    del buf317
    buf321 = buf274; del buf274  # reuse
    buf322 = buf273; del buf273  # reuse
    buf323 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf324 = reinterpret_tensor(buf319, (1536, 384, 1, 1), (384, 1, 1, 1), 0); del buf319  # reuse
    buf325 = buf318; del buf318  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_41(c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(unsqueeze_266.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()))
    del convolution_39
    del primals_91
    del primals_92
    del squeeze_61
    del unsqueeze_266
    # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act3], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.silu, aten.sub]
    buf326 = aten.convolution_backward(buf325, mul_133, view_89, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf325
    del mul_133
    del view_89
    buf327 = buf326[0]
    buf328 = buf326[1]
    buf329 = buf326[2]
    del buf326
    buf330 = buf301; del buf301  # reuse
    buf331 = buf300; del buf300  # reuse
    buf332 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf333 = reinterpret_tensor(buf289, (384, 64, 3, 3), (576, 9, 3, 1), 0); del buf289  # reuse
    buf334 = buf327; del buf327  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_42(c_void_p(buf334.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(unsqueeze_274.data_ptr()), c_void_p(squeeze_59.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()))
    del convolution_38
    del primals_88
    del primals_89
    del squeeze_59
    del unsqueeze_274
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf335 = aten.convolution_backward(buf334, mul_129, view_86, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf334
    del mul_129
    del view_86
    buf336 = buf335[0]
    buf337 = buf335[1]
    buf338 = buf335[2]
    del buf335
    buf339 = buf331; del buf331  # reuse
    buf340 = buf330; del buf330  # reuse
    buf341 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf342 = reinterpret_tensor(buf328, (384, 64, 3, 3), (576, 9, 3, 1), 0); del buf328  # reuse
    buf343 = buf336; del buf336  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_43(c_void_p(buf343.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(unsqueeze_282.data_ptr()), c_void_p(squeeze_57.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()))
    del convolution_37
    del primals_85
    del primals_86
    del squeeze_57
    del unsqueeze_282
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf344 = aten.convolution_backward(buf343, mul_125, view_83, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf343
    del mul_125
    del view_83
    buf345 = buf344[0]
    buf346 = buf344[1]
    buf347 = buf344[2]
    del buf344
    buf348 = buf340; del buf340  # reuse
    buf349 = buf339; del buf339  # reuse
    buf350 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf351 = reinterpret_tensor(buf346, (384, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf346  # reuse
    buf352 = reinterpret_tensor(buf313, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf313  # reuse
    buf353 = reinterpret_tensor(buf352, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf352  # reuse
    cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_44(c_void_p(buf351.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(unsqueeze_290.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(mul_705.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()))
    del convolution_34
    del primals_82
    del primals_83
    del squeeze_55
    del unsqueeze_290
    # Source Nodes: [sigmoid_4], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf354 = aten.convolution_backward(buf353, relu_4, primals_190, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf353
    del primals_190
    buf355 = buf354[0]
    buf356 = buf354[1]
    buf357 = buf354[2]
    del buf354
    buf358 = buf355; del buf355  # reuse
    cpp_fused_convolution_backward_threshold_backward_45(c_void_p(buf358.data_ptr()), c_void_p(relu_4.data_ptr()))
    del relu_4
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf359 = aten.convolution_backward(buf358, mean_4, primals_188, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf358
    del mean_4
    del primals_188
    buf360 = buf359[0]
    buf361 = buf359[1]
    buf362 = buf359[2]
    del buf359
    buf363 = buf316; del buf316  # reuse
    cpp_fused_add_convolution_backward_div_mul_sigmoid_46(c_void_p(buf304.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(mul_705.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf363.data_ptr()))
    del convolution_36
    # Source Nodes: [sigmoid_4], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf364 = aten.convolution_backward(buf363, mul_117, view_80, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf363
    del mul_117
    del view_80
    buf365 = buf364[0]
    buf366 = buf364[1]
    buf367 = buf364[2]
    del buf364
    buf368 = buf322; del buf322  # reuse
    buf369 = buf321; del buf321  # reuse
    buf370 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf371 = reinterpret_tensor(buf366, (1536, 384, 1, 1), (384, 1, 1, 1), 0); del buf366  # reuse
    buf372 = buf365; del buf365  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_47(c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(unsqueeze_298.data_ptr()), c_void_p(squeeze_53.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()))
    del convolution_33
    del primals_79
    del primals_80
    del squeeze_53
    del unsqueeze_298
    # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act3], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.silu, aten.sub]
    buf373 = aten.convolution_backward(buf372, mul_113, view_77, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf372
    del mul_113
    del view_77
    buf374 = buf373[0]
    buf375 = buf373[1]
    buf376 = buf373[2]
    del buf373
    buf377 = buf349; del buf349  # reuse
    buf378 = buf348; del buf348  # reuse
    buf379 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf380 = reinterpret_tensor(buf337, (384, 64, 3, 3), (576, 9, 3, 1), 0); del buf337  # reuse
    buf381 = buf374; del buf374  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_48(c_void_p(buf381.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(unsqueeze_306.data_ptr()), c_void_p(squeeze_51.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()))
    del convolution_32
    del primals_76
    del primals_77
    del squeeze_51
    del unsqueeze_306
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf382 = aten.convolution_backward(buf381, mul_109, view_74, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf381
    del mul_109
    del view_74
    buf383 = buf382[0]
    buf384 = buf382[1]
    buf385 = buf382[2]
    del buf382
    buf386 = buf378; del buf378  # reuse
    buf387 = buf377; del buf377  # reuse
    buf388 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf389 = reinterpret_tensor(buf375, (384, 64, 3, 3), (576, 9, 3, 1), 0); del buf375  # reuse
    buf390 = buf383; del buf383  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_49(c_void_p(buf390.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(unsqueeze_314.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()))
    del convolution_31
    del primals_73
    del primals_74
    del squeeze_49
    del unsqueeze_314
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf391 = aten.convolution_backward(buf390, mul_105, view_71, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf390
    del mul_105
    del view_71
    buf392 = buf391[0]
    buf393 = buf391[1]
    buf394 = buf391[2]
    del buf391
    buf395 = buf387; del buf387  # reuse
    buf396 = buf386; del buf386  # reuse
    buf397 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf398 = reinterpret_tensor(buf393, (384, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf393  # reuse
    buf399 = buf304; del buf304  # reuse
    buf400 = reinterpret_tensor(buf360, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf360  # reuse
    buf401 = reinterpret_tensor(buf400, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf400  # reuse
    cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_50(c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(unsqueeze_322.data_ptr()), c_void_p(squeeze_47.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(mul_705.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(mul_764.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf397.data_ptr()))
    del buf345
    del convolution_28
    del mul_705
    del mul_764
    del primals_70
    del primals_71
    del squeeze_47
    del unsqueeze_322
    # Source Nodes: [sigmoid_3], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf402 = aten.convolution_backward(buf401, relu_3, primals_186, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf401
    del primals_186
    buf403 = buf402[0]
    buf404 = buf402[1]
    buf405 = buf402[2]
    del buf402
    buf406 = buf403; del buf403  # reuse
    cpp_fused_convolution_backward_threshold_backward_51(c_void_p(buf406.data_ptr()), c_void_p(relu_3.data_ptr()))
    del relu_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf407 = aten.convolution_backward(buf406, mean_3, primals_184, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf406
    del mean_3
    del primals_184
    buf408 = buf407[0]
    buf409 = buf407[1]
    buf410 = buf407[2]
    del buf407
    buf411 = buf392; del buf392  # reuse
    cpp_fused_add_convolution_backward_div_mul_sigmoid_52(c_void_p(buf399.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf411.data_ptr()))
    del buf408
    del convolution_30
    # Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf412 = aten.convolution_backward(buf411, mul_97, view_68, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf411
    del mul_97
    del view_68
    buf413 = buf412[0]
    buf414 = buf412[1]
    buf415 = buf412[2]
    del buf412
    buf416 = buf369; del buf369  # reuse
    buf417 = buf368; del buf368  # reuse
    buf418 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf419 = reinterpret_tensor(buf414, (1536, 384, 1, 1), (384, 1, 1, 1), 0); del buf414  # reuse
    buf420 = buf413; del buf413  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_53(c_void_p(buf419.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(unsqueeze_330.data_ptr()), c_void_p(squeeze_45.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()))
    del convolution_27
    del primals_67
    del primals_68
    del squeeze_45
    del unsqueeze_330
    # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act3], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.silu, aten.sub]
    buf421 = aten.convolution_backward(buf420, mul_93, view_65, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf420
    del mul_93
    del view_65
    buf422 = buf421[0]
    buf423 = buf421[1]
    buf424 = buf421[2]
    del buf421
    buf425 = buf396; del buf396  # reuse
    buf426 = buf395; del buf395  # reuse
    buf427 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf428 = reinterpret_tensor(buf384, (384, 64, 3, 3), (576, 9, 3, 1), 0); del buf384  # reuse
    buf429 = buf422; del buf422  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_54(c_void_p(buf429.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(unsqueeze_338.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()))
    del convolution_26
    del primals_64
    del primals_65
    del squeeze_43
    del unsqueeze_338
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf430 = aten.convolution_backward(buf429, mul_89, view_62, [384], [2, 2], [1, 1], [1, 1], False, [0, 0], 6, [True, True, True])
    del buf429
    del mul_89
    del view_62
    buf431 = buf430[0]
    buf432 = buf430[1]
    buf433 = buf430[2]
    del buf430
    buf434 = buf426; del buf426  # reuse
    buf435 = buf425; del buf425  # reuse
    buf436 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf437 = reinterpret_tensor(buf423, (384, 64, 3, 3), (576, 9, 3, 1), 0); del buf423  # reuse
    buf438 = buf431; del buf431  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_55(c_void_p(buf438.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(unsqueeze_346.data_ptr()), c_void_p(squeeze_41.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()))
    del buf432
    del convolution_25
    del primals_61
    del primals_62
    del squeeze_41
    del unsqueeze_346
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf439 = aten.convolution_backward(buf438, mul_82, view_59, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf438
    del mul_82
    del view_59
    buf440 = buf439[0]
    buf441 = buf439[1]
    buf442 = buf439[2]
    del buf439
    buf443 = buf435; del buf435  # reuse
    buf444 = buf434; del buf434  # reuse
    buf445 = empty((384, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf446 = reinterpret_tensor(buf441, (384, 512, 1, 1), (512, 1, 1, 1), 0); del buf441  # reuse
    cpp_fused_mul_native_batch_norm_backward_view_56(c_void_p(buf446.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(unsqueeze_354.data_ptr()), c_void_p(squeeze_39.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()))
    del buf443
    del buf444
    del primals_58
    del primals_59
    del squeeze_39
    del unsqueeze_354
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf447 = aten.convolution_backward(buf399, avg_pool2d_1, view_56, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del avg_pool2d_1
    del buf399
    del view_56
    buf448 = buf447[0]
    buf449 = buf447[1]
    buf450 = buf447[2]
    del buf447
    buf451 = buf417; del buf417  # reuse
    buf452 = buf416; del buf416  # reuse
    buf453 = empty((1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf454 = reinterpret_tensor(buf449, (1536, 512, 1, 1), (512, 1, 1, 1), 0); del buf449  # reuse
    buf455 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cpu', dtype=torch.float32)
    buf456 = reinterpret_tensor(buf455, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf455  # reuse
    cpp_fused_add_avg_pool2d_backward_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_57(c_void_p(buf454.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(unsqueeze_362.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(mul_833.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()))
    del buf451
    del buf452
    del convolution_21
    del primals_55
    del primals_56
    del squeeze_37
    del unsqueeze_362
    # Source Nodes: [sigmoid_2], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf457 = aten.convolution_backward(buf456, relu_2, primals_182, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf456
    del primals_182
    buf458 = buf457[0]
    buf459 = buf457[1]
    buf460 = buf457[2]
    del buf457
    buf461 = buf458; del buf458  # reuse
    cpp_fused_convolution_backward_threshold_backward_58(c_void_p(buf461.data_ptr()), c_void_p(relu_2.data_ptr()))
    del relu_2
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf462 = aten.convolution_backward(buf461, mean_2, primals_180, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf461
    del mean_2
    del primals_180
    buf463 = buf462[0]
    buf464 = buf462[1]
    buf465 = buf462[2]
    del buf462
    buf466 = empty_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_avg_pool2d_backward_convolution_backward_div_mul_sigmoid_59(c_void_p(buf440.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(mul_833.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf466.data_ptr()))
    del convolution_23
    # Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.avg_pool2d_backward, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf467 = aten.convolution_backward(buf466, mul_74, view_53, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf466
    del mul_74
    del view_53
    buf468 = buf467[0]
    buf469 = buf467[1]
    buf470 = buf467[2]
    del buf467
    buf471 = empty((512, ), device='cpu', dtype=torch.float32)
    buf472 = empty((512, ), device='cpu', dtype=torch.float32)
    buf473 = empty((512, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf474 = reinterpret_tensor(buf469, (512, 128, 1, 1), (128, 1, 1, 1), 0); del buf469  # reuse
    buf475 = buf468; del buf468  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_60(c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(unsqueeze_370.data_ptr()), c_void_p(squeeze_35.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf473.data_ptr()))
    del convolution_20
    del primals_52
    del primals_53
    del squeeze_35
    del unsqueeze_370
    # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act3], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.silu, aten.sub]
    buf476 = aten.convolution_backward(buf475, mul_70, view_50, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, True])
    del buf475
    del mul_70
    del view_50
    buf477 = buf476[0]
    buf478 = buf476[1]
    buf479 = buf476[2]
    del buf476
    buf480 = empty((128, ), device='cpu', dtype=torch.float32)
    buf481 = empty((128, ), device='cpu', dtype=torch.float32)
    buf482 = empty((128, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf483 = empty((128, 64, 3, 3), device='cpu', dtype=torch.float32)
    buf484 = buf477; del buf477  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_61(c_void_p(buf484.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(unsqueeze_378.data_ptr()), c_void_p(squeeze_33.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()))
    del convolution_19
    del primals_49
    del primals_50
    del squeeze_33
    del unsqueeze_378
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf485 = aten.convolution_backward(buf484, mul_66, view_47, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, True])
    del buf484
    del mul_66
    del view_47
    buf486 = buf485[0]
    buf487 = buf485[1]
    buf488 = buf485[2]
    del buf485
    buf489 = buf481; del buf481  # reuse
    buf490 = buf480; del buf480  # reuse
    buf491 = empty((128, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf492 = reinterpret_tensor(buf478, (128, 64, 3, 3), (576, 9, 3, 1), 0); del buf478  # reuse
    buf493 = buf486; del buf486  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_62(c_void_p(buf493.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(unsqueeze_386.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf492.data_ptr()))
    del convolution_18
    del primals_46
    del primals_47
    del squeeze_31
    del unsqueeze_386
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf494 = aten.convolution_backward(buf493, mul_62, view_44, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf493
    del mul_62
    del view_44
    buf495 = buf494[0]
    buf496 = buf494[1]
    buf497 = buf494[2]
    del buf494
    buf498 = buf490; del buf490  # reuse
    buf499 = buf489; del buf489  # reuse
    buf500 = empty((128, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf501 = reinterpret_tensor(buf496, (128, 512, 1, 1), (512, 1, 1, 1), 0); del buf496  # reuse
    buf502 = buf440; del buf440  # reuse
    buf503 = reinterpret_tensor(buf463, (8, 512, 1, 1), (512, 1, 4096, 4096), 0); del buf463  # reuse
    buf504 = reinterpret_tensor(buf503, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf503  # reuse
    cpp_fused_add_avg_pool2d_backward_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_63(c_void_p(buf501.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(unsqueeze_394.data_ptr()), c_void_p(squeeze_29.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(mul_833.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(mul_892.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()))
    del buf448
    del convolution_15
    del mul_833
    del mul_892
    del primals_43
    del primals_44
    del squeeze_29
    del unsqueeze_394
    # Source Nodes: [sigmoid_1], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf505 = aten.convolution_backward(buf504, relu_1, primals_178, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf504
    del primals_178
    buf506 = buf505[0]
    buf507 = buf505[1]
    buf508 = buf505[2]
    del buf505
    buf509 = buf506; del buf506  # reuse
    cpp_fused_convolution_backward_threshold_backward_64(c_void_p(buf509.data_ptr()), c_void_p(relu_1.data_ptr()))
    del relu_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf510 = aten.convolution_backward(buf509, mean_1, primals_176, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf509
    del mean_1
    del primals_176
    buf511 = buf510[0]
    buf512 = buf510[1]
    buf513 = buf510[2]
    del buf510
    buf514 = buf495; del buf495  # reuse
    cpp_fused_add_convolution_backward_div_mul_sigmoid_65(c_void_p(buf502.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf514.data_ptr()))
    del buf511
    del convolution_17
    # Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf515 = aten.convolution_backward(buf514, mul_54, view_41, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf514
    del mul_54
    del view_41
    buf516 = buf515[0]
    buf517 = buf515[1]
    buf518 = buf515[2]
    del buf515
    buf519 = buf472; del buf472  # reuse
    buf520 = buf471; del buf471  # reuse
    buf521 = empty((512, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf522 = reinterpret_tensor(buf517, (512, 128, 1, 1), (128, 1, 1, 1), 0); del buf517  # reuse
    buf523 = buf516; del buf516  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_66(c_void_p(buf522.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(unsqueeze_402.data_ptr()), c_void_p(squeeze_27.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf521.data_ptr()))
    del convolution_14
    del primals_40
    del primals_41
    del squeeze_27
    del unsqueeze_402
    # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act3], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.silu, aten.sub]
    buf524 = aten.convolution_backward(buf523, mul_50, view_38, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, True])
    del buf523
    del mul_50
    del view_38
    buf525 = buf524[0]
    buf526 = buf524[1]
    buf527 = buf524[2]
    del buf524
    buf528 = buf499; del buf499  # reuse
    buf529 = buf498; del buf498  # reuse
    buf530 = empty((128, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf531 = reinterpret_tensor(buf487, (128, 64, 3, 3), (576, 9, 3, 1), 0); del buf487  # reuse
    buf532 = buf525; del buf525  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_67(c_void_p(buf532.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(unsqueeze_410.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf531.data_ptr()))
    del convolution_13
    del primals_37
    del primals_38
    del squeeze_25
    del unsqueeze_410
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf533 = aten.convolution_backward(buf532, mul_46, view_35, [128], [2, 2], [1, 1], [1, 1], False, [0, 0], 2, [True, True, True])
    del buf532
    del mul_46
    del view_35
    buf534 = buf533[0]
    buf535 = buf533[1]
    buf536 = buf533[2]
    del buf533
    buf537 = buf529; del buf529  # reuse
    buf538 = buf528; del buf528  # reuse
    buf539 = empty((128, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf540 = reinterpret_tensor(buf526, (128, 64, 3, 3), (576, 9, 3, 1), 0); del buf526  # reuse
    buf541 = buf534; del buf534  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_68(c_void_p(buf541.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(unsqueeze_418.data_ptr()), c_void_p(squeeze_23.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf540.data_ptr()))
    del convolution_12
    del primals_34
    del primals_35
    del squeeze_23
    del unsqueeze_418
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf542 = aten.convolution_backward(buf541, mul_39, view_32, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf541
    del mul_39
    del view_32
    buf543 = buf542[0]
    buf544 = buf542[1]
    buf545 = buf542[2]
    del buf542
    buf546 = buf538; del buf538  # reuse
    buf547 = buf537; del buf537  # reuse
    buf548 = empty((128, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf549 = reinterpret_tensor(buf544, (128, 256, 1, 1), (256, 1, 1, 1), 0); del buf544  # reuse
    cpp_fused_mul_native_batch_norm_backward_view_69(c_void_p(buf549.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(unsqueeze_426.data_ptr()), c_void_p(squeeze_21.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf548.data_ptr()))
    del primals_31
    del primals_32
    del squeeze_21
    del unsqueeze_426
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf550 = aten.convolution_backward(buf502, avg_pool2d, view_29, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del avg_pool2d
    del buf502
    del view_29
    buf551 = buf550[0]
    buf552 = buf550[1]
    buf553 = buf550[2]
    del buf550
    buf554 = buf520; del buf520  # reuse
    buf555 = buf519; del buf519  # reuse
    buf556 = empty((512, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf557 = reinterpret_tensor(buf552, (512, 256, 1, 1), (256, 1, 1, 1), 0); del buf552  # reuse
    buf558 = buf543; del buf543  # reuse
    buf559 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf560 = reinterpret_tensor(buf559, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf559  # reuse
    cpp_fused_add_avg_pool2d_backward_convolution_backward_mul_native_batch_norm_backward_sigmoid_sigmoid_backward_sum_view_70(c_void_p(buf557.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(unsqueeze_434.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(mul_961.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf555.data_ptr()), c_void_p(buf556.data_ptr()))
    del buf551
    del buf554
    del buf555
    del convolution_8
    del mul_961
    del primals_28
    del primals_29
    del squeeze_19
    del unsqueeze_434
    # Source Nodes: [sigmoid], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf561 = aten.convolution_backward(buf560, relu, primals_174, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf560
    del primals_174
    buf562 = buf561[0]
    buf563 = buf561[1]
    buf564 = buf561[2]
    del buf561
    buf565 = buf562; del buf562  # reuse
    cpp_fused_convolution_backward_threshold_backward_71(c_void_p(buf565.data_ptr()), c_void_p(relu.data_ptr()))
    del relu
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf566 = aten.convolution_backward(buf565, mean, primals_172, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf565
    del mean
    del primals_172
    buf567 = buf566[0]
    buf568 = buf566[1]
    buf569 = buf566[2]
    del buf566
    buf570 = empty_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_div_mul_sigmoid_72(c_void_p(buf558.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf570.data_ptr()))
    del buf567
    del convolution_10
    # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.sigmoid]
    buf571 = aten.convolution_backward(buf570, mul_31, view_26, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf570
    del mul_31
    del view_26
    buf572 = buf571[0]
    buf573 = buf571[1]
    buf574 = buf571[2]
    del buf571
    buf575 = empty((256, ), device='cpu', dtype=torch.float32)
    buf576 = empty((256, ), device='cpu', dtype=torch.float32)
    buf577 = empty((256, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf578 = reinterpret_tensor(buf573, (256, 64, 1, 1), (64, 1, 1, 1), 0); del buf573  # reuse
    buf579 = buf572; del buf572  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_73(c_void_p(buf578.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(unsqueeze_442.data_ptr()), c_void_p(squeeze_17.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf577.data_ptr()))
    del convolution_7
    del primals_25
    del primals_26
    del squeeze_17
    del unsqueeze_442
    # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act3], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.silu, aten.sub]
    buf580 = aten.convolution_backward(buf579, mul_27, view_23, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf579
    del mul_27
    del view_23
    buf581 = buf580[0]
    buf582 = buf580[1]
    buf583 = buf580[2]
    del buf580
    buf584 = empty((64, ), device='cpu', dtype=torch.float32)
    buf585 = empty((64, ), device='cpu', dtype=torch.float32)
    buf586 = empty((64, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf587 = empty((64, 64, 3, 3), device='cpu', dtype=torch.float32)
    buf588 = buf581; del buf581  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_74(c_void_p(buf588.data_ptr()), c_void_p(buf582.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(unsqueeze_450.data_ptr()), c_void_p(squeeze_15.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(buf587.data_ptr()))
    del convolution_6
    del primals_22
    del primals_23
    del squeeze_15
    del unsqueeze_450
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf589 = aten.convolution_backward(buf588, mul_23, view_20, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf588
    del mul_23
    del view_20
    buf590 = buf589[0]
    buf591 = buf589[1]
    buf592 = buf589[2]
    del buf589
    buf593 = buf585; del buf585  # reuse
    buf594 = buf584; del buf584  # reuse
    buf595 = empty((64, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf596 = reinterpret_tensor(buf582, (64, 64, 3, 3), (576, 9, 3, 1), 0); del buf582  # reuse
    buf597 = buf590; del buf590  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_75(c_void_p(buf597.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(unsqueeze_458.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(buf595.data_ptr()), c_void_p(buf596.data_ptr()))
    del buf591
    del convolution_5
    del primals_19
    del primals_20
    del squeeze_13
    del unsqueeze_458
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf598 = aten.convolution_backward(buf597, mul_16, view_17, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf597
    del view_17
    buf599 = buf598[0]
    buf600 = buf598[1]
    buf601 = buf598[2]
    del buf598
    buf602 = buf594; del buf594  # reuse
    buf603 = buf593; del buf593  # reuse
    buf604 = empty((64, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf605 = reinterpret_tensor(buf600, (64, 128, 1, 1), (128, 1, 1, 1), 0); del buf600  # reuse
    cpp_fused_mul_native_batch_norm_backward_view_76(c_void_p(buf605.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(unsqueeze_466.data_ptr()), c_void_p(squeeze_11.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf604.data_ptr()))
    del primals_16
    del primals_17
    del squeeze_11
    del unsqueeze_466
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf606 = aten.convolution_backward(buf558, mul_16, view_14, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf558
    del mul_16
    del view_14
    buf607 = buf606[0]
    buf608 = buf606[1]
    buf609 = buf606[2]
    del buf606
    buf610 = buf576; del buf576  # reuse
    buf611 = buf575; del buf575  # reuse
    buf612 = empty((256, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf613 = reinterpret_tensor(buf608, (256, 128, 1, 1), (128, 1, 1, 1), 0); del buf608  # reuse
    buf614 = buf599; del buf599  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_silu_sub_view_77(c_void_p(buf613.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(unsqueeze_474.data_ptr()), c_void_p(squeeze_9.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(buf610.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf612.data_ptr()))
    del buf607
    del buf610
    del buf611
    del convolution_3
    del primals_13
    del primals_14
    del squeeze_9
    del unsqueeze_474
    # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act1], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.silu, aten.sub]
    buf615 = aten.convolution_backward(buf614, mul_11, view_11, [128], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf614
    del mul_11
    del view_11
    buf616 = buf615[0]
    buf617 = buf615[1]
    buf618 = buf615[2]
    del buf615
    buf619 = buf547; del buf547  # reuse
    buf620 = buf546; del buf546  # reuse
    buf621 = empty((128, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf622 = reinterpret_tensor(buf535, (128, 64, 3, 3), (576, 9, 3, 1), 0); del buf535  # reuse
    buf623 = buf616; del buf616  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_78(c_void_p(buf623.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(unsqueeze_482.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(buf621.data_ptr()), c_void_p(buf622.data_ptr()))
    del buf617
    del buf619
    del buf620
    del convolution_2
    del primals_10
    del primals_11
    del squeeze_7
    del unsqueeze_482
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf624 = aten.convolution_backward(buf623, mul_7, view_8, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf623
    del mul_7
    del view_8
    buf625 = buf624[0]
    buf626 = buf624[1]
    buf627 = buf624[2]
    del buf624
    buf628 = buf603; del buf603  # reuse
    buf629 = buf602; del buf602  # reuse
    buf630 = empty((64, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf631 = reinterpret_tensor(buf0, (64, 32, 3, 3), (288, 9, 3, 1), 0); del buf0  # reuse
    buf632 = buf625; del buf625  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_79(c_void_p(buf632.data_ptr()), c_void_p(buf626.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(unsqueeze_490.data_ptr()), c_void_p(squeeze_5.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(buf628.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(buf631.data_ptr()))
    del buf626
    del buf628
    del buf629
    del convolution_1
    del primals_7
    del primals_8
    del squeeze_5
    del unsqueeze_490
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf633 = aten.convolution_backward(buf632, mul_3, view_5, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf632
    del mul_3
    del view_5
    buf634 = buf633[0]
    buf635 = buf633[1]
    buf636 = buf633[2]
    del buf633
    buf637 = empty((32, ), device='cpu', dtype=torch.float32)
    buf638 = empty((32, ), device='cpu', dtype=torch.float32)
    buf639 = empty((32, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf640 = empty((32, 16, 3, 3), device='cpu', dtype=torch.float32)
    buf641 = buf634; del buf634  # reuse
    cpp_fused_add_convolution_backward_fill_mul_native_batch_norm_backward_sigmoid_sub_view_80(c_void_p(buf641.data_ptr()), c_void_p(buf635.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(unsqueeze_498.data_ptr()), c_void_p(squeeze_3.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(buf637.data_ptr()), c_void_p(buf638.data_ptr()), c_void_p(buf639.data_ptr()), c_void_p(buf640.data_ptr()))
    del buf635
    del buf637
    del buf638
    del convolution
    del primals_4
    del primals_5
    del squeeze_3
    del unsqueeze_498
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.fill, aten.mul, aten.sigmoid, aten.sub]
    buf642 = aten.convolution_backward(buf641, primals_222, view_2, [16], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf641
    del primals_222
    del view_2
    buf643 = buf642[1]
    buf644 = buf642[2]
    del buf642
    buf645 = empty((16, ), device='cpu', dtype=torch.float32)
    buf646 = empty((16, ), device='cpu', dtype=torch.float32)
    buf647 = empty((16, 1, 1, 1), device='cpu', dtype=torch.float32)
    buf648 = empty((16, 3, 3, 3), device='cpu', dtype=torch.float32)
    cpp_fused_mul_native_batch_norm_backward_view_81(c_void_p(buf643.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(unsqueeze_506.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf645.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf647.data_ptr()), c_void_p(buf648.data_ptr()))
    del buf643
    del buf645
    del buf646
    del primals_1
    del primals_2
    del squeeze_1
    del unsqueeze_506
    return (buf648, buf647, buf644, buf640, buf639, buf636, buf631, buf630, buf627, buf622, buf621, buf618, buf613, buf612, buf609, buf605, buf604, buf601, buf596, buf595, buf592, buf587, buf586, buf583, buf578, buf577, buf574, buf557, buf556, buf553, buf549, buf548, buf545, buf540, buf539, buf536, buf531, buf530, buf527, buf522, buf521, buf518, buf501, buf500, buf497, buf492, buf491, buf488, buf483, buf482, buf479, buf474, buf473, buf470, buf454, buf453, buf450, buf446, buf445, buf442, buf437, buf436, buf433, buf428, buf427, buf424, buf419, buf418, buf415, buf398, buf397, buf394, buf389, buf388, buf385, buf380, buf379, buf376, buf371, buf370, buf367, buf351, buf350, buf347, buf342, buf341, buf338, buf333, buf332, buf329, buf324, buf323, buf320, buf303, buf302, buf299, buf294, buf293, buf290, buf285, buf284, buf281, buf276, buf275, buf272, buf256, buf255, buf252, buf247, buf246, buf243, buf238, buf237, buf234, buf229, buf228, buf225, buf208, buf207, buf204, buf199, buf198, buf195, buf190, buf189, buf186, buf181, buf180, buf177, buf161, buf160, buf157, buf153, buf152, buf149, buf144, buf143, buf140, buf135, buf134, buf131, buf126, buf125, buf122, buf105, buf104, buf101, buf96, buf95, buf92, buf87, buf86, buf83, buf78, buf77, buf74, buf58, buf57, buf54, buf49, buf48, buf45, buf40, buf39, buf36, buf31, buf30, buf27, buf11, buf10, buf7, buf568, buf569, buf563, buf564, buf512, buf513, buf507, buf508, buf464, buf465, buf459, buf460, buf409, buf410, buf404, buf405, buf361, buf362, buf356, buf357, buf314, buf315, buf309, buf310, buf266, buf267, buf261, buf262, buf219, buf220, buf214, buf215, buf171, buf172, buf166, buf167, buf116, buf117, buf111, buf112, buf68, buf69, buf63, buf64, buf21, buf22, buf16, buf17, reinterpret_tensor(buf1, (1000, 2304), (2304, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((32, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((384, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((1536, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((2304, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((2304, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    view_2 = rand_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    mul_3 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    squeeze_3 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    view_5 = rand_strided((32, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    mul_7 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    squeeze_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    view_8 = rand_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    mul_11 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_11 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    mul_16 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    squeeze_9 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_14 = rand_strided((256, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    squeeze_11 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    view_17 = rand_strided((64, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    mul_23 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    view_20 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    mul_27 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    squeeze_15 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    view_23 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    mul_31 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    squeeze_17 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    view_26 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    mean = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    relu = rand_strided((8, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    mul_39 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    avg_pool2d = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    view_29 = rand_strided((512, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    squeeze_21 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_32 = rand_strided((128, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    mul_46 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    squeeze_23 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_35 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    mul_50 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_38 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    mul_54 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_27 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    view_41 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    mean_1 = rand_strided((8, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((8, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    mul_62 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    squeeze_29 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_44 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    mul_66 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_47 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    mul_70 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_33 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    view_50 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    mul_74 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    squeeze_35 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    view_53 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    mean_2 = rand_strided((8, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((8, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    mul_82 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    avg_pool2d_1 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_56 = rand_strided((1536, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    squeeze_39 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_59 = rand_strided((384, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    mul_89 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    squeeze_41 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_62 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    mul_93 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_65 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    mul_97 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_45 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_68 = rand_strided((1536, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    mean_3 = rand_strided((8, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((8, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    mul_105 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    squeeze_47 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_71 = rand_strided((384, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    mul_109 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_74 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    mul_113 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_51 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_77 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    mul_117 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_53 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_80 = rand_strided((1536, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    mean_4 = rand_strided((8, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((8, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    mul_125 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_83 = rand_strided((384, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    mul_129 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_57 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_86 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    mul_133 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_59 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_89 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    mul_137 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_92 = rand_strided((1536, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    mean_5 = rand_strided((8, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((8, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((8, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    mul_145 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    squeeze_63 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_95 = rand_strided((384, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    mul_149 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_65 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_98 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    mul_153 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_101 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    mul_157 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_69 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_104 = rand_strided((1536, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    mean_6 = rand_strided((8, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((8, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((8, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    mul_165 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    squeeze_71 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_107 = rand_strided((384, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    mul_169 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_110 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    mul_173 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_75 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_113 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    mul_177 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_77 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_116 = rand_strided((1536, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    mean_7 = rand_strided((8, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((8, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    convolution_54 = rand_strided((8, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    mul_185 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_119 = rand_strided((384, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    mul_189 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_81 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_122 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_56 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    mul_193 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_83 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_125 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_57 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    mul_197 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_128 = rand_strided((1536, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    convolution_58 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    mean_8 = rand_strided((8, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((8, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    convolution_60 = rand_strided((8, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    mul_205 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    avg_pool2d_2 = rand_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    squeeze_87 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_131 = rand_strided((1536, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    squeeze_89 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_134 = rand_strided((384, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    convolution_62 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    mul_212 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_137 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_63 = rand_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    mul_216 = rand_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    squeeze_93 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_140 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_64 = rand_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    mul_220 = rand_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    squeeze_95 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_143 = rand_strided((1536, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    convolution_65 = rand_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    mean_9 = rand_strided((8, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((8, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    convolution_67 = rand_strided((8, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    mul_228 = rand_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_146 = rand_strided((384, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    convolution_68 = rand_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    mul_232 = rand_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    squeeze_99 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_149 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_69 = rand_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    mul_236 = rand_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    squeeze_101 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_152 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_70 = rand_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    mul_240 = rand_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_155 = rand_strided((1536, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    convolution_71 = rand_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    mean_10 = rand_strided((8, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((8, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    convolution_73 = rand_strided((8, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    mul_248 = rand_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    squeeze_105 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_158 = rand_strided((384, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    convolution_74 = rand_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    mul_252 = rand_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    squeeze_107 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_161 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_75 = rand_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    mul_256 = rand_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    view_164 = rand_strided((384, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    convolution_76 = rand_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    mul_260 = rand_strided((8, 384, 7, 7), (18816, 1, 2688, 384), device='cpu', dtype=torch.float32)
    squeeze_111 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    view_167 = rand_strided((1536, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    convolution_77 = rand_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    mean_11 = rand_strided((8, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((8, 384, 1, 1), (384, 1, 384, 384), device='cpu', dtype=torch.float32)
    convolution_79 = rand_strided((8, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    add_67 = rand_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    squeeze_113 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    view_170 = rand_strided((2304, 1536, 1, 1), (1536, 1, 1536, 1536), device='cpu', dtype=torch.float32)
    convolution_80 = rand_strided((8, 2304, 7, 7), (112896, 1, 16128, 2304), device='cpu', dtype=torch.float32)
    clone_28 = rand_strided((8, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    unsqueeze_58 = rand_strided((1, 2304, 1), (2304, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_66 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_74 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_82 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_90 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    mul_341 = rand_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    unsqueeze_98 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_106 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_114 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_122 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    mul_400 = rand_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cpu', dtype=torch.float32)
    unsqueeze_130 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_138 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_146 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_154 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_162 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    mul_469 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    unsqueeze_170 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_178 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_186 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_194 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    mul_528 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    unsqueeze_202 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_210 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_218 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_226 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    mul_587 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    unsqueeze_234 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_242 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    mul_646 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    unsqueeze_266 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_274 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_282 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_290 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    mul_705 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_306 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_314 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    mul_764 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_338 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_354 = rand_strided((1, 384, 1), (384, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_362 = rand_strided((1, 1536, 1), (1536, 1, 1), device='cpu', dtype=torch.float32)
    mul_833 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    unsqueeze_370 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_378 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_386 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    mul_892 = rand_strided((8, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_410 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_418 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_426 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_434 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    mul_961 = rand_strided((8, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 256, 1), (256, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_450 = rand_strided((1, 64, 1), (64, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_458 = rand_strided((1, 64, 1), (64, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_466 = rand_strided((1, 64, 1), (64, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 256, 1), (256, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_482 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 64, 1), (64, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_498 = rand_strided((1, 32, 1), (32, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_506 = rand_strided((1, 16, 1), (16, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_174, primals_176, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, primals_204, primals_206, primals_208, primals_210, primals_212, primals_214, primals_216, primals_218, primals_222, squeeze_1, view_2, convolution, mul_3, squeeze_3, view_5, convolution_1, mul_7, squeeze_5, view_8, convolution_2, mul_11, squeeze_7, view_11, convolution_3, mul_16, squeeze_9, view_14, squeeze_11, view_17, convolution_5, mul_23, squeeze_13, view_20, convolution_6, mul_27, squeeze_15, view_23, convolution_7, mul_31, squeeze_17, view_26, convolution_8, mean, relu, convolution_10, mul_39, avg_pool2d, squeeze_19, view_29, squeeze_21, view_32, convolution_12, mul_46, squeeze_23, view_35, convolution_13, mul_50, squeeze_25, view_38, convolution_14, mul_54, squeeze_27, view_41, convolution_15, mean_1, relu_1, convolution_17, mul_62, squeeze_29, view_44, convolution_18, mul_66, squeeze_31, view_47, convolution_19, mul_70, squeeze_33, view_50, convolution_20, mul_74, squeeze_35, view_53, convolution_21, mean_2, relu_2, convolution_23, mul_82, avg_pool2d_1, squeeze_37, view_56, squeeze_39, view_59, convolution_25, mul_89, squeeze_41, view_62, convolution_26, mul_93, squeeze_43, view_65, convolution_27, mul_97, squeeze_45, view_68, convolution_28, mean_3, relu_3, convolution_30, mul_105, squeeze_47, view_71, convolution_31, mul_109, squeeze_49, view_74, convolution_32, mul_113, squeeze_51, view_77, convolution_33, mul_117, squeeze_53, view_80, convolution_34, mean_4, relu_4, convolution_36, mul_125, squeeze_55, view_83, convolution_37, mul_129, squeeze_57, view_86, convolution_38, mul_133, squeeze_59, view_89, convolution_39, mul_137, squeeze_61, view_92, convolution_40, mean_5, relu_5, convolution_42, mul_145, squeeze_63, view_95, convolution_43, mul_149, squeeze_65, view_98, convolution_44, mul_153, squeeze_67, view_101, convolution_45, mul_157, squeeze_69, view_104, convolution_46, mean_6, relu_6, convolution_48, mul_165, squeeze_71, view_107, convolution_49, mul_169, squeeze_73, view_110, convolution_50, mul_173, squeeze_75, view_113, convolution_51, mul_177, squeeze_77, view_116, convolution_52, mean_7, relu_7, convolution_54, mul_185, squeeze_79, view_119, convolution_55, mul_189, squeeze_81, view_122, convolution_56, mul_193, squeeze_83, view_125, convolution_57, mul_197, squeeze_85, view_128, convolution_58, mean_8, relu_8, convolution_60, mul_205, avg_pool2d_2, squeeze_87, view_131, squeeze_89, view_134, convolution_62, mul_212, squeeze_91, view_137, convolution_63, mul_216, squeeze_93, view_140, convolution_64, mul_220, squeeze_95, view_143, convolution_65, mean_9, relu_9, convolution_67, mul_228, squeeze_97, view_146, convolution_68, mul_232, squeeze_99, view_149, convolution_69, mul_236, squeeze_101, view_152, convolution_70, mul_240, squeeze_103, view_155, convolution_71, mean_10, relu_10, convolution_73, mul_248, squeeze_105, view_158, convolution_74, mul_252, squeeze_107, view_161, convolution_75, mul_256, squeeze_109, view_164, convolution_76, mul_260, squeeze_111, view_167, convolution_77, mean_11, relu_11, convolution_79, add_67, squeeze_113, view_170, convolution_80, clone_28, permute_1, unsqueeze_58, unsqueeze_66, unsqueeze_74, unsqueeze_82, unsqueeze_90, mul_341, unsqueeze_98, unsqueeze_106, unsqueeze_114, unsqueeze_122, mul_400, unsqueeze_130, unsqueeze_138, unsqueeze_146, unsqueeze_154, unsqueeze_162, mul_469, unsqueeze_170, unsqueeze_178, unsqueeze_186, unsqueeze_194, mul_528, unsqueeze_202, unsqueeze_210, unsqueeze_218, unsqueeze_226, mul_587, unsqueeze_234, unsqueeze_242, unsqueeze_250, unsqueeze_258, mul_646, unsqueeze_266, unsqueeze_274, unsqueeze_282, unsqueeze_290, mul_705, unsqueeze_298, unsqueeze_306, unsqueeze_314, unsqueeze_322, mul_764, unsqueeze_330, unsqueeze_338, unsqueeze_346, unsqueeze_354, unsqueeze_362, mul_833, unsqueeze_370, unsqueeze_378, unsqueeze_386, unsqueeze_394, mul_892, unsqueeze_402, unsqueeze_410, unsqueeze_418, unsqueeze_426, unsqueeze_434, mul_961, unsqueeze_442, unsqueeze_450, unsqueeze_458, unsqueeze_466, unsqueeze_474, unsqueeze_482, unsqueeze_490, unsqueeze_498, unsqueeze_506, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('nfnet_l0', benchmark_compiled_module)
