
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


cpp_fused_convolution_backward_sum_threshold_backward_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
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
            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(0.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
            tmp4.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_1 = async_compile.cpp('''
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (960L*x2) + (47040L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (960L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (960L*x2) + (47040L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(960L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
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
                        tmp25.store(out_ptr3 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_as_strided_scatter_convolution_backward_copy_native_batch_norm_backward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (160L*x2) + (7840L*x0)), static_cast<long>(160L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x2) + (7840L*x0)));
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(62720L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(62720L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(62720L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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
                        at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(3920L + x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(3920L + x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x2) + (80L*x2_inner) + (3920L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp1 * tmp4;
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                            tmp_acc1_vec = tmp_acc1_vec + tmp5;
                        }
                    }
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr2[static_cast<long>(3920L + x2 + (49L*x0) + (49L*x0_inner) + (7840L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x2) + (3920L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                }
                tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr5 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)), static_cast<long>(80L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(3920L + x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                        auto tmp6 = out_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp9 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp15 = out_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp19 = in_ptr4[static_cast<long>(x1 + x1_inner)];
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
                        tmp22.store(out_ptr6 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(3920L + x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
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
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_3 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x2) + (80L*x2_inner) + (3920L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x2) + (80L*x2_inner) + (3920L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp4 - tmp5;
                            auto tmp7 = tmp3 * tmp6;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                            tmp_acc1_vec = tmp_acc1_vec + tmp7;
                        }
                    }
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (49L*x0) + (49L*x0_inner) + (7840L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x2) + (3920L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x2) + (3920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)), static_cast<long>(80L), tmp1, 8);
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)), static_cast<long>(80L), tmp4, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                        auto tmp6 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp12 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                        auto tmp22 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp10 = static_cast<float>(0.002551020408163265);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp13 = decltype(tmp12)(tmp12 * tmp12);
                        auto tmp14 = decltype(tmp11)(tmp11 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp8 * tmp15;
                        auto tmp17 = tmp3 - tmp16;
                        auto tmp19 = decltype(tmp18)(tmp18 * tmp10);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp17 - tmp20;
                        auto tmp23 = decltype(tmp12)(tmp12 * tmp22);
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp21 * tmp24;
                        tmp25.store(out_ptr3 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)));
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
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp20.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_threshold_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2)
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
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
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
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (480L*x2) + (23520L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(480L + x0 + (960L*x2) + (47040L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(480L + x0 + (960L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(480L + x0 + (960L*x1)));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (480L*x2) + (23520L*x1)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp5 = static_cast<float>(49.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = tmp3 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp15 = tmp11 * tmp14;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                            tmp_acc1_vec = tmp_acc1_vec + tmp15;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (480L*x1) + (23520L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(480L + x2 + (960L*x1) + (47040L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(480L + x2 + (960L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(480L + x2 + (960L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (480L*x1) + (23520L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = static_cast<float>(49.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp8 = tmp3 + tmp7;
                        auto tmp9 = static_cast<float>(0.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                        auto tmp14 = tmp12 - tmp13;
                        auto tmp16 = static_cast<float>(0.002551020408163265);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 * tmp17;
                        auto tmp20 = tmp19 * tmp19;
                        auto tmp21 = tmp18 * tmp20;
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp23 = tmp11 - tmp22;
                        auto tmp25 = tmp24 * tmp17;
                        auto tmp26 = tmp23 - tmp25;
                        tmp26.store(out_ptr2 + static_cast<long>(x2 + (480L*x1) + (23520L*x0)));
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
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr1 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_7 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (480L*x1) + (23520L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (23520L*x0)));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = static_cast<float>(49.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 / tmp9;
                        auto tmp11 = tmp6 + tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = decltype(tmp2)::blendv(tmp13, tmp2, tmp3);
                        tmp14.store(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (23520L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (480L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (480L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_8 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (160L*x2) + (7840L*x0)), static_cast<long>(160L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (160L*x2) + (7840L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(62720L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(62720L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(62720L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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
                        at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(3920L + x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(3920L + x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x2) + (80L*x2_inner) + (3920L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp1 * tmp4;
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                            tmp_acc1_vec = tmp_acc1_vec + tmp5;
                        }
                    }
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr2[static_cast<long>(3920L + x2 + (49L*x0) + (49L*x0_inner) + (7840L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x2) + (3920L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                }
                tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr5 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)), static_cast<long>(80L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(3920L + x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp6 = out_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp9 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp15 = out_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp19 = in_ptr5[static_cast<long>(x1 + x1_inner)];
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
                        tmp22.store(out_ptr6 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(3920L + x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_9 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x2) + (80L*x2_inner) + (3920L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x2) + (80L*x2_inner) + (3920L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp4 - tmp5;
                            auto tmp7 = tmp3 * tmp6;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                            tmp_acc1_vec = tmp_acc1_vec + tmp7;
                        }
                    }
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (49L*x0) + (49L*x0_inner) + (7840L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x2) + (3920L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x2) + (3920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)), static_cast<long>(80L), tmp1, 8);
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)), static_cast<long>(80L), tmp4, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                        auto tmp6 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp12 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                        auto tmp22 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp10 = static_cast<float>(0.002551020408163265);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp13 = decltype(tmp12)(tmp12 * tmp12);
                        auto tmp14 = decltype(tmp11)(tmp11 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp8 * tmp15;
                        auto tmp17 = tmp3 - tmp16;
                        auto tmp19 = decltype(tmp18)(tmp18 * tmp10);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp17 - tmp20;
                        auto tmp23 = decltype(tmp12)(tmp12 * tmp22);
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp21 * tmp24;
                        tmp25.store(out_ptr3 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)));
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
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp20.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(480L + x0 + (960L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(480L + x1 + (960L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (480L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_11 = async_compile.cpp('''
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
    auto in_ptr2 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (480L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (480L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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


cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_12 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (160L*x2) + (7840L*x0)), static_cast<long>(160L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (160L*x2) + (7840L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(62720L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(62720L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(62720L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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
                        at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(3920L + x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(3920L + x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x2) + (80L*x2_inner) + (3920L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp1 * tmp4;
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                            tmp_acc1_vec = tmp_acc1_vec + tmp5;
                        }
                    }
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr2[static_cast<long>(3920L + x2 + (49L*x0) + (49L*x0_inner) + (7840L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x2) + (3920L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                }
                tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr5 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)), static_cast<long>(80L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(3920L + x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp6 = out_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp9 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp15 = out_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp19 = in_ptr5[static_cast<long>(x1 + x1_inner)];
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
                        tmp22.store(out_ptr6 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(3920L + x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_13 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x2) + (80L*x2_inner) + (3920L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x2) + (80L*x2_inner) + (3920L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp4 - tmp5;
                            auto tmp7 = tmp3 * tmp6;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                            tmp_acc1_vec = tmp_acc1_vec + tmp7;
                        }
                    }
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (49L*x0) + (49L*x0_inner) + (7840L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x2) + (3920L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x2) + (3920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)), static_cast<long>(80L), tmp1, 8);
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)), static_cast<long>(80L), tmp4, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                        auto tmp6 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp12 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                        auto tmp22 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp10 = static_cast<float>(0.002551020408163265);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp13 = decltype(tmp12)(tmp12 * tmp12);
                        auto tmp14 = decltype(tmp11)(tmp11 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp8 * tmp15;
                        auto tmp17 = tmp3 - tmp16;
                        auto tmp19 = decltype(tmp18)(tmp18 * tmp10);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp17 - tmp20;
                        auto tmp23 = decltype(tmp12)(tmp12 * tmp22);
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp21 * tmp24;
                        tmp25.store(out_ptr3 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)));
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
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp20.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_threshold_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2)
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
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
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
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_16 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (480L*x2) + (23520L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(480L + x0 + (960L*x2) + (47040L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(480L + x0 + (960L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(480L + x0 + (960L*x1)));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (480L*x2) + (23520L*x1)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp5 = static_cast<float>(49.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = tmp3 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp15 = tmp11 * tmp14;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                            tmp_acc1_vec = tmp_acc1_vec + tmp15;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (480L*x1) + (23520L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(480L + x2 + (960L*x1) + (47040L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(480L + x2 + (960L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(480L + x2 + (960L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (480L*x1) + (23520L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = static_cast<float>(49.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp8 = tmp3 + tmp7;
                        auto tmp9 = static_cast<float>(0.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                        auto tmp14 = tmp12 - tmp13;
                        auto tmp16 = static_cast<float>(0.002551020408163265);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 * tmp17;
                        auto tmp20 = tmp19 * tmp19;
                        auto tmp21 = tmp18 * tmp20;
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp23 = tmp11 - tmp22;
                        auto tmp25 = tmp24 * tmp17;
                        auto tmp26 = tmp23 - tmp25;
                        tmp26.store(out_ptr2 + static_cast<long>(x2 + (480L*x1) + (23520L*x0)));
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
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr1 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_17 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (480L*x1) + (23520L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (23520L*x0)));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = static_cast<float>(49.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 / tmp9;
                        auto tmp11 = tmp6 + tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = decltype(tmp2)::blendv(tmp13, tmp2, tmp3);
                        tmp14.store(in_out_ptr0 + static_cast<long>(x2 + (480L*x1) + (23520L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (480L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (480L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_18 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (160L*x2) + (7840L*x0)), static_cast<long>(160L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (160L*x2) + (7840L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(62720L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(62720L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(62720L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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
                        at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(3920L + x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(3920L + x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x2) + (80L*x2_inner) + (3920L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp1 * tmp4;
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                            tmp_acc1_vec = tmp_acc1_vec + tmp5;
                        }
                    }
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr2[static_cast<long>(3920L + x2 + (49L*x0) + (49L*x0_inner) + (7840L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x2) + (3920L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                }
                tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr5 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)), static_cast<long>(80L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(3920L + x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp6 = out_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp9 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp15 = out_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp19 = in_ptr5[static_cast<long>(x1 + x1_inner)];
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
                        tmp22.store(out_ptr6 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(3920L + x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0))] = tmpbuf[x1_inner]; }
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
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x2) + (80L*x2_inner) + (3920L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x2) + (80L*x2_inner) + (3920L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp4 - tmp5;
                            auto tmp7 = tmp3 * tmp6;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                            tmp_acc1_vec = tmp_acc1_vec + tmp7;
                        }
                    }
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (49L*x0) + (49L*x0_inner) + (7840L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x2) + (3920L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x2) + (3920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)), static_cast<long>(80L), tmp1, 8);
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)), static_cast<long>(80L), tmp4, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                        auto tmp6 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp12 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                        auto tmp22 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp10 = static_cast<float>(0.002551020408163265);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp13 = decltype(tmp12)(tmp12 * tmp12);
                        auto tmp14 = decltype(tmp11)(tmp11 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp8 * tmp15;
                        auto tmp17 = tmp3 - tmp16;
                        auto tmp19 = decltype(tmp18)(tmp18 * tmp10);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp17 - tmp20;
                        auto tmp23 = decltype(tmp12)(tmp12 * tmp22);
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp21 * tmp24;
                        tmp25.store(out_ptr3 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)));
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
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp20.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(480L + x0 + (960L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(480L + x1 + (960L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (480L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_21 = async_compile.cpp('''
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
    auto in_ptr2 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (480L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (480L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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


cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_22 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (160L*x2) + (7840L*x0)), static_cast<long>(160L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (160L*x2) + (7840L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(62720L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(62720L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(62720L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            tmp0.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
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
                        at::vec::transpose_mxn<float,8,8>(out_ptr1 + static_cast<long>(x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(out_ptr1 + static_cast<long>(x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x2) + (160L*x2_inner) + (7840L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp1 * tmp4;
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                            tmp_acc1_vec = tmp_acc1_vec + tmp5;
                        }
                    }
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr1[static_cast<long>(x2 + (49L*x0) + (49L*x0_inner) + (7840L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x2) + (7840L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                }
                tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr5 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (160L*x2) + (7840L*x0)), static_cast<long>(160L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp6 = out_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp9 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp15 = out_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp19 = in_ptr5[static_cast<long>(x1 + x1_inner)];
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
                        tmp22.store(out_ptr6 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x2) + (7840L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_23 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
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
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (112L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (112L*x0)));
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
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (112L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_24 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(3920L + x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(3920L + x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x2) + (80L*x2_inner) + (3920L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp1 * tmp4;
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                            tmp_acc1_vec = tmp_acc1_vec + tmp5;
                        }
                    }
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(3920L + x2 + (49L*x0) + (49L*x0_inner) + (7840L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x2) + (3920L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)), static_cast<long>(80L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(3920L + x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp9 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp15 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                        auto tmp19 = in_ptr4[static_cast<long>(x1 + x1_inner)];
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
                        tmp22.store(out_ptr3 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(3920L + x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)));
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
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0))] = tmpbuf[x1_inner]; }
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (49L*x0) + (7840L*x1)), static_cast<long>(49L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x2) + (80L*x2_inner) + (3920L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x2) + (80L*x2_inner) + (3920L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp4 - tmp5;
                            auto tmp7 = tmp3 * tmp6;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                            tmp_acc1_vec = tmp_acc1_vec + tmp7;
                        }
                    }
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (49L*x0) + (49L*x0_inner) + (7840L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x2) + (3920L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x2) + (3920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)), static_cast<long>(80L), tmp1, 8);
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)), static_cast<long>(80L), tmp4, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                        auto tmp6 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp12 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                        auto tmp22 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp10 = static_cast<float>(0.002551020408163265);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp13 = decltype(tmp12)(tmp12 * tmp12);
                        auto tmp14 = decltype(tmp11)(tmp11 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp8 * tmp15;
                        auto tmp17 = tmp3 - tmp16;
                        auto tmp19 = decltype(tmp18)(tmp18 * tmp10);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp17 - tmp20;
                        auto tmp23 = decltype(tmp12)(tmp12 * tmp22);
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp21 * tmp24;
                        tmp25.store(out_ptr3 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7840L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (80L*x2) + (3920L*x0)));
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
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp20.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (3920L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_threshold_backward_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2)
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
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
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
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1344L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mul_native_batch_norm_backward_28 = async_compile.cpp('''
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
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (672L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (672L*x2) + (32928L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = static_cast<float>(49.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = tmp2 + tmp6;
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = static_cast<float>(49.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 + tmp6;
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
                        tmp25.store(in_out_ptr0 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (336L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(336L + x0 + (672L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (336L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (336L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(336L + x1 + (672L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (336L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (336L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_30 = async_compile.cpp('''
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
    auto in_ptr2 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (336L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (672L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (336L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (336L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (336L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (336L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (336L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (336L*x0)));
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
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_as_strided_scatter_convolution_backward_copy_native_batch_norm_backward_31 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(112L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (112L*x1) + (112L*x1_inner) + (21952L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (112L*x1) + (112L*x1_inner) + (21952L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (21952L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(112L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (112L*x1) + (21952L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (112L*x1) + (21952L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (21952L*x0))] = tmp2;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(175616L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(175616L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(175616L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
                                at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(10976L + x2 + (196L*x0) + (21952L*x1)), static_cast<long>(196L), tmp0, 8);
                                at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(10976L + x2 + (196L*x0) + (21952L*x1)), static_cast<long>(196L), tmp0, 8);
                                for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (56L*x2) + (56L*x2_inner) + (10976L*x1)));
                                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                    auto tmp4 = tmp2 - tmp3;
                                    auto tmp5 = tmp1 * tmp4;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                    tmp_acc1_vec = tmp_acc1_vec + tmp5;
                                }
                            }
                            for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr2[static_cast<long>(10976L + x2 + (196L*x0) + (196L*x0_inner) + (21952L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (56L*x2) + (10976L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp3 = tmp1 - tmp2;
                                auto tmp4 = tmp0 * tmp3;
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                                tmp_acc1_vec = tmp_acc1_vec + tmp4;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                        tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (56L*x2) + (10976L*x0)), static_cast<long>(56L), tmp1, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(10976L + x2 + (196L*x1) + (196L*x1_inner) + (21952L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                                auto tmp3 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                                auto tmp6 = out_ptr4[static_cast<long>(x1 + x1_inner)];
                                auto tmp9 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                                auto tmp15 = out_ptr3[static_cast<long>(x1 + x1_inner)];
                                auto tmp19 = in_ptr5[static_cast<long>(x1 + x1_inner)];
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
                                tmp22.store(out_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (10976L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(10976L + x2 + (196L*x1) + (196L*x1_inner) + (21952L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (56L*x2) + (10976L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (10976L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_32 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (21952L*x1)), static_cast<long>(196L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (21952L*x1)), static_cast<long>(196L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (56L*x2) + (56L*x2_inner) + (10976L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (56L*x2) + (56L*x2_inner) + (10976L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp4 - tmp5;
                            auto tmp7 = tmp3 * tmp6;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                            tmp_acc1_vec = tmp_acc1_vec + tmp7;
                        }
                    }
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (21952L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (56L*x2) + (10976L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (56L*x2) + (10976L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (56L*x2) + (10976L*x0)), static_cast<long>(56L), tmp1, 8);
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (56L*x2) + (10976L*x0)), static_cast<long>(56L), tmp4, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (21952L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                        auto tmp6 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp12 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                        auto tmp22 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp10 = static_cast<float>(0.0006377551020408163);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp13 = decltype(tmp12)(tmp12 * tmp12);
                        auto tmp14 = decltype(tmp11)(tmp11 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp8 * tmp15;
                        auto tmp17 = tmp3 - tmp16;
                        auto tmp19 = decltype(tmp18)(tmp18 * tmp10);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp17 - tmp20;
                        auto tmp23 = decltype(tmp12)(tmp12 * tmp22);
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp21 * tmp24;
                        tmp25.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (10976L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (21952L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (56L*x2) + (10976L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (56L*x2) + (10976L*x0)));
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
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp20.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (10976L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_threshold_backward_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2)
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
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
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
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1344L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_35 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (336L*x2) + (65856L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(336L + x0 + (672L*x2) + (131712L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(336L + x0 + (672L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(336L + x0 + (672L*x1)));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (336L*x2) + (65856L*x1)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp5 = static_cast<float>(196.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = tmp3 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp15 = tmp11 * tmp14;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                            tmp_acc1_vec = tmp_acc1_vec + tmp15;
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (336L*x1) + (65856L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(336L + x2 + (672L*x1) + (131712L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(336L + x2 + (672L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(336L + x2 + (672L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (336L*x1) + (65856L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = static_cast<float>(196.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp8 = tmp3 + tmp7;
                        auto tmp9 = static_cast<float>(0.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                        auto tmp14 = tmp12 - tmp13;
                        auto tmp16 = static_cast<float>(0.0006377551020408163);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 * tmp17;
                        auto tmp20 = tmp19 * tmp19;
                        auto tmp21 = tmp18 * tmp20;
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp23 = tmp11 - tmp22;
                        auto tmp25 = tmp24 * tmp17;
                        auto tmp26 = tmp23 - tmp25;
                        tmp26.store(out_ptr2 + static_cast<long>(x2 + (336L*x1) + (65856L*x0)));
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
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (336L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr1 + static_cast<long>(x1 + (336L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_36 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(336L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (336L*x1) + (65856L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (336L*x1) + (65856L*x0)));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = static_cast<float>(196.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 / tmp9;
                        auto tmp11 = tmp6 + tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = decltype(tmp2)::blendv(tmp13, tmp2, tmp3);
                        tmp14.store(in_out_ptr0 + static_cast<long>(x2 + (336L*x1) + (65856L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (336L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (336L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (336L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (336L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (336L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_37 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (112L*x2) + (21952L*x0)), static_cast<long>(112L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (21952L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (21952L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (21952L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (112L*x2) + (21952L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (21952L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(175616L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(175616L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(175616L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr1 + static_cast<long>(x2 + (196L*x0) + (21952L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr1 + static_cast<long>(x2 + (196L*x0) + (21952L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (112L*x2) + (112L*x2_inner) + (21952L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr1[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (21952L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (112L*x2) + (21952L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (112L*x2) + (21952L*x0)), static_cast<long>(112L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (21952L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr5[static_cast<long>(x1 + x1_inner)];
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
                            tmp22.store(out_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (21952L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (21952L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (112L*x2) + (21952L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (21952L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_38 = async_compile.cpp('''
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_39 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(10976L + x2 + (196L*x0) + (21952L*x1)), static_cast<long>(196L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(10976L + x2 + (196L*x0) + (21952L*x1)), static_cast<long>(196L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (56L*x2) + (56L*x2_inner) + (10976L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp1 * tmp4;
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                            tmp_acc1_vec = tmp_acc1_vec + tmp5;
                        }
                    }
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(10976L + x2 + (196L*x0) + (196L*x0_inner) + (21952L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (56L*x2) + (10976L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (56L*x2) + (10976L*x0)), static_cast<long>(56L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(10976L + x2 + (196L*x1) + (196L*x1_inner) + (21952L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp9 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp15 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                        auto tmp19 = in_ptr4[static_cast<long>(x1 + x1_inner)];
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
                        tmp22.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (10976L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(10976L + x2 + (196L*x1) + (196L*x1_inner) + (21952L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (56L*x2) + (10976L*x0)));
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
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (10976L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_40 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (21952L*x1)), static_cast<long>(196L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (21952L*x1)), static_cast<long>(196L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (56L*x2) + (56L*x2_inner) + (10976L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (56L*x2) + (56L*x2_inner) + (10976L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp4 - tmp5;
                            auto tmp7 = tmp3 * tmp6;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                            tmp_acc1_vec = tmp_acc1_vec + tmp7;
                        }
                    }
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (21952L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (56L*x2) + (10976L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (56L*x2) + (10976L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (56L*x2) + (10976L*x0)), static_cast<long>(56L), tmp1, 8);
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (56L*x2) + (10976L*x0)), static_cast<long>(56L), tmp4, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (21952L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                        auto tmp6 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp12 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                        auto tmp22 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp10 = static_cast<float>(0.0006377551020408163);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp13 = decltype(tmp12)(tmp12 * tmp12);
                        auto tmp14 = decltype(tmp11)(tmp11 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp8 * tmp15;
                        auto tmp17 = tmp3 - tmp16;
                        auto tmp19 = decltype(tmp18)(tmp18 * tmp10);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp17 - tmp20;
                        auto tmp23 = decltype(tmp12)(tmp12 * tmp22);
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp21 * tmp24;
                        tmp25.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (10976L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (21952L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (56L*x2) + (10976L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (56L*x2) + (10976L*x0)));
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
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp20.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (10976L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_threshold_backward_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2)
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
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
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
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (240L*x2) + (47040L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(240L + x0 + (480L*x2) + (94080L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(240L + x0 + (480L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(240L + x0 + (480L*x1)));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (240L*x2) + (47040L*x1)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp5 = static_cast<float>(196.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = tmp3 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp15 = tmp11 * tmp14;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                            tmp_acc1_vec = tmp_acc1_vec + tmp15;
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (240L*x1) + (47040L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(240L + x2 + (480L*x1) + (94080L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(240L + x2 + (480L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(240L + x2 + (480L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (240L*x1) + (47040L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = static_cast<float>(196.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp8 = tmp3 + tmp7;
                        auto tmp9 = static_cast<float>(0.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                        auto tmp14 = tmp12 - tmp13;
                        auto tmp16 = static_cast<float>(0.0006377551020408163);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 * tmp17;
                        auto tmp20 = tmp19 * tmp19;
                        auto tmp21 = tmp18 * tmp20;
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp23 = tmp11 - tmp22;
                        auto tmp25 = tmp24 * tmp17;
                        auto tmp26 = tmp23 - tmp25;
                        tmp26.store(out_ptr2 + static_cast<long>(x2 + (240L*x1) + (47040L*x0)));
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
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr1 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_44 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (240L*x1) + (47040L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (47040L*x0)));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = static_cast<float>(196.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 / tmp9;
                        auto tmp11 = tmp6 + tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = decltype(tmp2)::blendv(tmp13, tmp2, tmp3);
                        tmp14.store(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (47040L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (240L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (240L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_as_strided_scatter_convolution_backward_copy_native_batch_norm_backward_45 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x1) + (80L*x1_inner) + (15680L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x1) + (80L*x1_inner) + (15680L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (15680L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (80L*x1) + (15680L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (80L*x1) + (15680L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (15680L*x0))] = tmp2;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(125440L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(125440L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(125440L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
                                at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(7840L + x2 + (196L*x0) + (15680L*x1)), static_cast<long>(196L), tmp0, 8);
                                at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(7840L + x2 + (196L*x0) + (15680L*x1)), static_cast<long>(196L), tmp0, 8);
                                for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x2) + (40L*x2_inner) + (7840L*x1)));
                                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                    auto tmp4 = tmp2 - tmp3;
                                    auto tmp5 = tmp1 * tmp4;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                    tmp_acc1_vec = tmp_acc1_vec + tmp5;
                                }
                            }
                            for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr2[static_cast<long>(7840L + x2 + (196L*x0) + (196L*x0_inner) + (15680L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x2) + (7840L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp3 = tmp1 - tmp2;
                                auto tmp4 = tmp0 * tmp3;
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                                tmp_acc1_vec = tmp_acc1_vec + tmp4;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                        tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)), static_cast<long>(40L), tmp1, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(7840L + x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                                auto tmp3 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                                auto tmp6 = out_ptr4[static_cast<long>(x1 + x1_inner)];
                                auto tmp9 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                                auto tmp15 = out_ptr3[static_cast<long>(x1 + x1_inner)];
                                auto tmp19 = in_ptr5[static_cast<long>(x1 + x1_inner)];
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
                                tmp22.store(out_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (7840L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(7840L + x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (7840L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (15680L*x1)), static_cast<long>(196L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (15680L*x1)), static_cast<long>(196L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (40L*x2) + (40L*x2_inner) + (7840L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x2) + (40L*x2_inner) + (7840L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp4 - tmp5;
                            auto tmp7 = tmp3 * tmp6;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                            tmp_acc1_vec = tmp_acc1_vec + tmp7;
                        }
                    }
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (15680L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (40L*x2) + (7840L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x2) + (7840L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)), static_cast<long>(40L), tmp1, 8);
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)), static_cast<long>(40L), tmp4, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                        auto tmp6 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp12 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                        auto tmp22 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp10 = static_cast<float>(0.0006377551020408163);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp13 = decltype(tmp12)(tmp12 * tmp12);
                        auto tmp14 = decltype(tmp11)(tmp11 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp8 * tmp15;
                        auto tmp17 = tmp3 - tmp16;
                        auto tmp19 = decltype(tmp18)(tmp18 * tmp10);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp17 - tmp20;
                        auto tmp23 = decltype(tmp12)(tmp12 * tmp22);
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp21 * tmp24;
                        tmp25.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (7840L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)));
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
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp20.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (7840L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (92L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(92L + x0 + (184L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (92L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (92L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(92L + x0 + (184L*x1))];
                        auto tmp4 = in_ptr2[static_cast<long>(x0 + (92L*x1))];
                        auto tmp5 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = tmp0 ? tmp2 : tmp1;
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp3;
                        tmp_acc1 = tmp_acc1 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(88L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (92L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(92L + x1 + (184L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (92L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (92L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(88L); x1<static_cast<long>(92L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (92L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(92L + x1 + (184L*x0))];
                    auto tmp4 = in_ptr2[static_cast<long>(x1 + (92L*x0))];
                    auto tmp5 = in_ptr3[static_cast<long>(x1)];
                    auto tmp7 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr4[static_cast<long>(x1)];
                    auto tmp15 = out_ptr0[static_cast<long>(x1)];
                    auto tmp18 = in_ptr5[static_cast<long>(x1)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = tmp0 ? tmp2 : tmp1;
                    auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                    auto tmp8 = static_cast<float>(0.0006377551020408163);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp11 = decltype(tmp10)(tmp10 * tmp10);
                    auto tmp12 = decltype(tmp9)(tmp9 * tmp11);
                    auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                    auto tmp14 = decltype(tmp3)(tmp3 - tmp13);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                    auto tmp17 = decltype(tmp14)(tmp14 - tmp16);
                    auto tmp19 = decltype(tmp10)(tmp10 * tmp18);
                    auto tmp20 = decltype(tmp17)(tmp17 * tmp19);
                    out_ptr3[static_cast<long>(x1 + (92L*x0))] = tmp20;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_48 = async_compile.cpp('''
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
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (92L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (184L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (92L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (92L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (92L*x1))];
                        auto tmp3 = in_ptr1[static_cast<long>(x0 + (184L*x1))];
                        auto tmp4 = in_ptr2[static_cast<long>(x0 + (92L*x1))];
                        auto tmp7 = in_ptr3[static_cast<long>(x0 + (92L*x1))];
                        auto tmp8 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp6 = tmp2 ? tmp1 : tmp5;
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        tmp_acc0 = tmp_acc0 + tmp6;
                        tmp_acc1 = tmp_acc1 + tmp10;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(88L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (92L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (92L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (92L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (92L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(88L); x1<static_cast<long>(92L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (92L*x0))];
                    auto tmp3 = in_ptr1[static_cast<long>(x1 + (184L*x0))];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x1 + (92L*x0))];
                    auto tmp7 = in_ptr3[static_cast<long>(x1 + (92L*x0))];
                    auto tmp8 = in_ptr4[static_cast<long>(x1)];
                    auto tmp10 = out_ptr1[static_cast<long>(x1)];
                    auto tmp13 = in_ptr5[static_cast<long>(x1)];
                    auto tmp18 = out_ptr0[static_cast<long>(x1)];
                    auto tmp21 = in_ptr6[static_cast<long>(x1)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = tmp2 ? tmp1 : tmp5;
                    auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                    auto tmp11 = static_cast<float>(0.0006377551020408163);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp13)(tmp13 * tmp13);
                    auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                    auto tmp16 = decltype(tmp9)(tmp9 * tmp15);
                    auto tmp17 = decltype(tmp6)(tmp6 - tmp16);
                    auto tmp19 = decltype(tmp18)(tmp18 * tmp11);
                    auto tmp20 = decltype(tmp17)(tmp17 - tmp19);
                    auto tmp22 = decltype(tmp13)(tmp13 * tmp21);
                    auto tmp23 = decltype(tmp20)(tmp20 * tmp22);
                    in_out_ptr0[static_cast<long>(x1 + (92L*x0))] = tmp23;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    in_out_ptr1[static_cast<long>(x0)] = tmp2;
                }
            }
        }
    }
}
''')


cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_49 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (15680L*x0)), static_cast<long>(80L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (15680L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(125440L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(125440L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(125440L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
                                at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(7840L + x2 + (196L*x0) + (15680L*x1)), static_cast<long>(196L), tmp0, 8);
                                at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(7840L + x2 + (196L*x0) + (15680L*x1)), static_cast<long>(196L), tmp0, 8);
                                for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x2) + (40L*x2_inner) + (7840L*x1)));
                                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                    auto tmp4 = tmp2 - tmp3;
                                    auto tmp5 = tmp1 * tmp4;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                    tmp_acc1_vec = tmp_acc1_vec + tmp5;
                                }
                            }
                            for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr2[static_cast<long>(7840L + x2 + (196L*x0) + (196L*x0_inner) + (15680L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x2) + (7840L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp3 = tmp1 - tmp2;
                                auto tmp4 = tmp0 * tmp3;
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                                tmp_acc1_vec = tmp_acc1_vec + tmp4;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                        tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)), static_cast<long>(40L), tmp1, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(7840L + x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                                auto tmp3 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                                auto tmp6 = out_ptr4[static_cast<long>(x1 + x1_inner)];
                                auto tmp9 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                                auto tmp15 = out_ptr3[static_cast<long>(x1 + x1_inner)];
                                auto tmp19 = in_ptr5[static_cast<long>(x1 + x1_inner)];
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
                                tmp22.store(out_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (7840L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(7840L + x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (7840L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_50 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (15680L*x1)), static_cast<long>(196L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (15680L*x1)), static_cast<long>(196L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (40L*x2) + (40L*x2_inner) + (7840L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x2) + (40L*x2_inner) + (7840L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp4 - tmp5;
                            auto tmp7 = tmp3 * tmp6;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                            tmp_acc1_vec = tmp_acc1_vec + tmp7;
                        }
                    }
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (15680L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (40L*x2) + (7840L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x2) + (7840L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)), static_cast<long>(40L), tmp1, 8);
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)), static_cast<long>(40L), tmp4, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                        auto tmp6 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp12 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                        auto tmp22 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp10 = static_cast<float>(0.0006377551020408163);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp13 = decltype(tmp12)(tmp12 * tmp12);
                        auto tmp14 = decltype(tmp11)(tmp11 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp8 * tmp15;
                        auto tmp17 = tmp3 - tmp16;
                        auto tmp19 = decltype(tmp18)(tmp18 * tmp10);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp17 - tmp20;
                        auto tmp23 = decltype(tmp12)(tmp12 * tmp22);
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp21 * tmp24;
                        tmp25.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (7840L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)));
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
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp20.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (7840L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (92L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(92L + x0 + (184L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (92L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (92L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(92L + x0 + (184L*x1))];
                        auto tmp4 = in_ptr2[static_cast<long>(x0 + (92L*x1))];
                        auto tmp5 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = tmp0 ? tmp2 : tmp1;
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp3;
                        tmp_acc1 = tmp_acc1 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(88L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (92L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(92L + x1 + (184L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (92L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (92L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(88L); x1<static_cast<long>(92L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (92L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(92L + x1 + (184L*x0))];
                    auto tmp4 = in_ptr2[static_cast<long>(x1 + (92L*x0))];
                    auto tmp5 = in_ptr3[static_cast<long>(x1)];
                    auto tmp7 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr4[static_cast<long>(x1)];
                    auto tmp15 = out_ptr0[static_cast<long>(x1)];
                    auto tmp18 = in_ptr5[static_cast<long>(x1)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = tmp0 ? tmp2 : tmp1;
                    auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                    auto tmp8 = static_cast<float>(0.0006377551020408163);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp11 = decltype(tmp10)(tmp10 * tmp10);
                    auto tmp12 = decltype(tmp9)(tmp9 * tmp11);
                    auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                    auto tmp14 = decltype(tmp3)(tmp3 - tmp13);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                    auto tmp17 = decltype(tmp14)(tmp14 - tmp16);
                    auto tmp19 = decltype(tmp10)(tmp10 * tmp18);
                    auto tmp20 = decltype(tmp17)(tmp17 * tmp19);
                    out_ptr3[static_cast<long>(x1 + (92L*x0))] = tmp20;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_52 = async_compile.cpp('''
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
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (92L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (184L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (92L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (92L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (92L*x1))];
                        auto tmp3 = in_ptr1[static_cast<long>(x0 + (184L*x1))];
                        auto tmp4 = in_ptr2[static_cast<long>(x0 + (92L*x1))];
                        auto tmp7 = in_ptr3[static_cast<long>(x0 + (92L*x1))];
                        auto tmp8 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp6 = tmp2 ? tmp1 : tmp5;
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        tmp_acc0 = tmp_acc0 + tmp6;
                        tmp_acc1 = tmp_acc1 + tmp10;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(88L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (92L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (92L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (92L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (92L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(88L); x1<static_cast<long>(92L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (92L*x0))];
                    auto tmp3 = in_ptr1[static_cast<long>(x1 + (184L*x0))];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x1 + (92L*x0))];
                    auto tmp7 = in_ptr3[static_cast<long>(x1 + (92L*x0))];
                    auto tmp8 = in_ptr4[static_cast<long>(x1)];
                    auto tmp10 = out_ptr1[static_cast<long>(x1)];
                    auto tmp13 = in_ptr5[static_cast<long>(x1)];
                    auto tmp18 = out_ptr0[static_cast<long>(x1)];
                    auto tmp21 = in_ptr6[static_cast<long>(x1)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = tmp2 ? tmp1 : tmp5;
                    auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                    auto tmp11 = static_cast<float>(0.0006377551020408163);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp13)(tmp13 * tmp13);
                    auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                    auto tmp16 = decltype(tmp9)(tmp9 * tmp15);
                    auto tmp17 = decltype(tmp6)(tmp6 - tmp16);
                    auto tmp19 = decltype(tmp18)(tmp18 * tmp11);
                    auto tmp20 = decltype(tmp17)(tmp17 - tmp19);
                    auto tmp22 = decltype(tmp13)(tmp13 * tmp21);
                    auto tmp23 = decltype(tmp20)(tmp20 * tmp22);
                    in_out_ptr0[static_cast<long>(x1 + (92L*x0))] = tmp23;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    in_out_ptr1[static_cast<long>(x0)] = tmp2;
                }
            }
        }
    }
}
''')


cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_53 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (15680L*x0)), static_cast<long>(80L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (15680L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(125440L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(125440L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(125440L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
                                at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(7840L + x2 + (196L*x0) + (15680L*x1)), static_cast<long>(196L), tmp0, 8);
                                at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(7840L + x2 + (196L*x0) + (15680L*x1)), static_cast<long>(196L), tmp0, 8);
                                for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x2) + (40L*x2_inner) + (7840L*x1)));
                                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                    auto tmp4 = tmp2 - tmp3;
                                    auto tmp5 = tmp1 * tmp4;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                    tmp_acc1_vec = tmp_acc1_vec + tmp5;
                                }
                            }
                            for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr2[static_cast<long>(7840L + x2 + (196L*x0) + (196L*x0_inner) + (15680L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x2) + (7840L*x1)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp3 = tmp1 - tmp2;
                                auto tmp4 = tmp0 * tmp3;
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                                tmp_acc1_vec = tmp_acc1_vec + tmp4;
                            }
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                        tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)), static_cast<long>(40L), tmp1, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(7840L + x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                                auto tmp3 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                                auto tmp6 = out_ptr4[static_cast<long>(x1 + x1_inner)];
                                auto tmp9 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                                auto tmp15 = out_ptr3[static_cast<long>(x1 + x1_inner)];
                                auto tmp19 = in_ptr5[static_cast<long>(x1 + x1_inner)];
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
                                tmp22.store(out_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (7840L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(7840L + x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (7840L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_54 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (15680L*x1)), static_cast<long>(196L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (15680L*x1)), static_cast<long>(196L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (40L*x2) + (40L*x2_inner) + (7840L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x2) + (40L*x2_inner) + (7840L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp4 - tmp5;
                            auto tmp7 = tmp3 * tmp6;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                            tmp_acc1_vec = tmp_acc1_vec + tmp7;
                        }
                    }
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (15680L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (40L*x2) + (7840L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x2) + (7840L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)), static_cast<long>(40L), tmp1, 8);
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)), static_cast<long>(40L), tmp4, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                        auto tmp6 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp12 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                        auto tmp22 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp10 = static_cast<float>(0.0006377551020408163);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp13 = decltype(tmp12)(tmp12 * tmp12);
                        auto tmp14 = decltype(tmp11)(tmp11 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp8 * tmp15;
                        auto tmp17 = tmp3 - tmp16;
                        auto tmp19 = decltype(tmp18)(tmp18 * tmp10);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp17 - tmp20;
                        auto tmp23 = decltype(tmp12)(tmp12 * tmp22);
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp21 * tmp24;
                        tmp25.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (7840L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)));
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
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp20.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (7840L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_55 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (100L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(100L + x0 + (200L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (100L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(96L); x0<static_cast<long>(100L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (100L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(100L + x0 + (200L*x1))];
                        auto tmp4 = in_ptr2[static_cast<long>(x0 + (100L*x1))];
                        auto tmp5 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = tmp0 ? tmp2 : tmp1;
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp3;
                        tmp_acc1 = tmp_acc1 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(96L); x0<static_cast<long>(100L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (100L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(100L + x1 + (200L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (100L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (100L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(96L); x1<static_cast<long>(100L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (100L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(100L + x1 + (200L*x0))];
                    auto tmp4 = in_ptr2[static_cast<long>(x1 + (100L*x0))];
                    auto tmp5 = in_ptr3[static_cast<long>(x1)];
                    auto tmp7 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr4[static_cast<long>(x1)];
                    auto tmp15 = out_ptr0[static_cast<long>(x1)];
                    auto tmp18 = in_ptr5[static_cast<long>(x1)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = tmp0 ? tmp2 : tmp1;
                    auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                    auto tmp8 = static_cast<float>(0.0006377551020408163);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp11 = decltype(tmp10)(tmp10 * tmp10);
                    auto tmp12 = decltype(tmp9)(tmp9 * tmp11);
                    auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                    auto tmp14 = decltype(tmp3)(tmp3 - tmp13);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                    auto tmp17 = decltype(tmp14)(tmp14 - tmp16);
                    auto tmp19 = decltype(tmp10)(tmp10 * tmp18);
                    auto tmp20 = decltype(tmp17)(tmp17 * tmp19);
                    out_ptr3[static_cast<long>(x1 + (100L*x0))] = tmp20;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_56 = async_compile.cpp('''
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
    auto in_ptr2 = in_out_ptr0;
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (100L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (100L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (100L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(96L); x0<static_cast<long>(100L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (100L*x1))];
                        auto tmp3 = in_ptr1[static_cast<long>(x0 + (200L*x1))];
                        auto tmp4 = in_ptr2[static_cast<long>(x0 + (100L*x1))];
                        auto tmp7 = in_ptr3[static_cast<long>(x0 + (100L*x1))];
                        auto tmp8 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp6 = tmp2 ? tmp1 : tmp5;
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        tmp_acc0 = tmp_acc0 + tmp6;
                        tmp_acc1 = tmp_acc1 + tmp10;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (100L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (100L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(96L); x1<static_cast<long>(100L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (100L*x0))];
                    auto tmp3 = in_ptr1[static_cast<long>(x1 + (200L*x0))];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x1 + (100L*x0))];
                    auto tmp7 = in_ptr3[static_cast<long>(x1 + (100L*x0))];
                    auto tmp8 = in_ptr4[static_cast<long>(x1)];
                    auto tmp10 = out_ptr1[static_cast<long>(x1)];
                    auto tmp13 = in_ptr5[static_cast<long>(x1)];
                    auto tmp18 = out_ptr0[static_cast<long>(x1)];
                    auto tmp21 = in_ptr6[static_cast<long>(x1)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = tmp2 ? tmp1 : tmp5;
                    auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                    auto tmp11 = static_cast<float>(0.0006377551020408163);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp13)(tmp13 * tmp13);
                    auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                    auto tmp16 = decltype(tmp9)(tmp9 * tmp15);
                    auto tmp17 = decltype(tmp6)(tmp6 - tmp16);
                    auto tmp19 = decltype(tmp18)(tmp18 * tmp11);
                    auto tmp20 = decltype(tmp17)(tmp17 - tmp19);
                    auto tmp22 = decltype(tmp13)(tmp13 * tmp21);
                    auto tmp23 = decltype(tmp20)(tmp20 * tmp22);
                    in_out_ptr0[static_cast<long>(x1 + (100L*x0))] = tmp23;
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(96L); x0<static_cast<long>(100L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    in_out_ptr1[static_cast<long>(x0)] = tmp2;
                }
            }
        }
    }
}
''')


cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_57 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (15680L*x0)), static_cast<long>(80L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x2) + (15680L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(125440L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(125440L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(125440L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr1 + static_cast<long>(x2 + (196L*x0) + (15680L*x1)), static_cast<long>(196L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr1 + static_cast<long>(x2 + (196L*x0) + (15680L*x1)), static_cast<long>(196L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x2) + (80L*x2_inner) + (15680L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr1[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (15680L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (80L*x2) + (15680L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (80L*x2) + (15680L*x0)), static_cast<long>(80L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr5[static_cast<long>(x1 + x1_inner)];
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
                            tmp22.store(out_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (80L*x2) + (15680L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_58 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (40L*x0)));
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
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (40L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_59 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(7840L + x2 + (196L*x0) + (15680L*x1)), static_cast<long>(196L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(7840L + x2 + (196L*x0) + (15680L*x1)), static_cast<long>(196L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (40L*x2) + (40L*x2_inner) + (7840L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                            auto tmp4 = tmp2 - tmp3;
                            auto tmp5 = tmp1 * tmp4;
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                            tmp_acc1_vec = tmp_acc1_vec + tmp5;
                        }
                    }
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(7840L + x2 + (196L*x0) + (196L*x0_inner) + (15680L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (40L*x2) + (7840L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)), static_cast<long>(40L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(7840L + x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp9 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp15 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                        auto tmp19 = in_ptr4[static_cast<long>(x1 + x1_inner)];
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
                        tmp22.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (7840L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(7840L + x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)));
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
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (7840L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_60 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (15680L*x1)), static_cast<long>(196L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (196L*x0) + (15680L*x1)), static_cast<long>(196L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (40L*x2) + (40L*x2_inner) + (7840L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x2) + (40L*x2_inner) + (7840L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp6 = tmp4 - tmp5;
                            auto tmp7 = tmp3 * tmp6;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                            tmp_acc1_vec = tmp_acc1_vec + tmp7;
                        }
                    }
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (15680L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (40L*x2) + (7840L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x2) + (7840L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)), static_cast<long>(40L), tmp1, 8);
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)), static_cast<long>(40L), tmp4, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                        auto tmp6 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                        auto tmp12 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                        auto tmp22 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp10 = static_cast<float>(0.0006377551020408163);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp13 = decltype(tmp12)(tmp12 * tmp12);
                        auto tmp14 = decltype(tmp11)(tmp11 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp8 * tmp15;
                        auto tmp17 = tmp3 - tmp16;
                        auto tmp19 = decltype(tmp18)(tmp18 * tmp10);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp17 - tmp20;
                        auto tmp23 = decltype(tmp12)(tmp12 * tmp22);
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp21 * tmp24;
                        tmp25.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (7840L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (15680L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (40L*x2) + (7840L*x0)));
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
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp20.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (7840L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (240L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(120L + x0 + (240L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(120L + x1 + (240L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (120L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_63 = async_compile.cpp('''
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
    auto in_ptr2 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (120L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (120L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
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
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_as_strided_scatter_convolution_backward_copy_native_batch_norm_backward_64 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(40L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (40L*x1) + (40L*x1_inner) + (31360L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (40L*x1) + (40L*x1_inner) + (31360L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr0 + static_cast<long>(x1 + (784L*x2) + (31360L*x0)), static_cast<long>(784L));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(250880L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(250880L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(250880L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(15680L + x2 + (784L*x0) + (31360L*x1)), static_cast<long>(784L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(15680L + x2 + (784L*x0) + (31360L*x1)), static_cast<long>(784L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (20L*x2) + (20L*x2_inner) + (15680L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
            #pragma omp for 
            for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr2[static_cast<long>(15680L + x2 + (784L*x0) + (31360L*x1))];
                            auto tmp1 = in_ptr2[static_cast<long>(x0 + (20L*x2) + (15680L*x1))];
                            auto tmp2 = in_ptr3[static_cast<long>(x0)];
                            auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp0;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                    }
                    out_ptr3[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr4[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr4[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr5[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (20L*x2) + (15680L*x0)), static_cast<long>(20L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(15680L + x2 + (784L*x1) + (784L*x1_inner) + (31360L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr5[static_cast<long>(x1 + x1_inner)];
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
                            tmp22.store(out_ptr6 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (15680L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(16L); x1<static_cast<long>(20L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(15680L + x2 + (784L*x1) + (31360L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x1 + (20L*x2) + (15680L*x0))];
                        auto tmp2 = in_ptr3[static_cast<long>(x1)];
                        auto tmp4 = out_ptr4[static_cast<long>(x1)];
                        auto tmp7 = in_ptr4[static_cast<long>(x1)];
                        auto tmp12 = out_ptr3[static_cast<long>(x1)];
                        auto tmp15 = in_ptr5[static_cast<long>(x1)];
                        auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                        auto tmp5 = static_cast<float>(0.00015943877551020407);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp7)(tmp7 * tmp7);
                        auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                        auto tmp10 = decltype(tmp3)(tmp3 * tmp9);
                        auto tmp11 = decltype(tmp0)(tmp0 - tmp10);
                        auto tmp13 = decltype(tmp12)(tmp12 * tmp5);
                        auto tmp14 = decltype(tmp11)(tmp11 - tmp13);
                        auto tmp16 = decltype(tmp7)(tmp7 * tmp15);
                        auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                        out_ptr6[static_cast<long>(x2 + (784L*x1) + (15680L*x0))] = tmp17;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (784L*x0) + (31360L*x1)), static_cast<long>(784L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (784L*x0) + (31360L*x1)), static_cast<long>(784L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (20L*x2) + (20L*x2_inner) + (15680L*x1)));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (20L*x2) + (20L*x2_inner) + (15680L*x1)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp4 - tmp5;
                                auto tmp7 = tmp3 * tmp6;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp7;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for 
            for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (784L*x0) + (31360L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (20L*x2) + (15680L*x1))];
                            auto tmp3 = in_ptr2[static_cast<long>(x0 + (20L*x2) + (15680L*x1))];
                            auto tmp4 = in_ptr3[static_cast<long>(x0)];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                            auto tmp6 = decltype(tmp2)(tmp2 * tmp5);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp6;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (20L*x2) + (15680L*x0)), static_cast<long>(20L), tmp1, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (20L*x2) + (15680L*x0)), static_cast<long>(20L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (31360L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp12 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp18 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                            auto tmp22 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(0.00015943877551020407);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp13 = decltype(tmp12)(tmp12 * tmp12);
                            auto tmp14 = decltype(tmp11)(tmp11 * tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            auto tmp17 = tmp3 - tmp16;
                            auto tmp19 = decltype(tmp18)(tmp18 * tmp10);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp17 - tmp20;
                            auto tmp23 = decltype(tmp12)(tmp12 * tmp22);
                            auto tmp24 = at::vec::Vectorized<float>(tmp23);
                            auto tmp25 = tmp21 * tmp24;
                            tmp25.store(out_ptr3 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (15680L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(16L); x1<static_cast<long>(20L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (784L*x1) + (31360L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (20L*x2) + (15680L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x1 + (20L*x2) + (15680L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp6 = out_ptr1[static_cast<long>(x1)];
                        auto tmp9 = in_ptr4[static_cast<long>(x1)];
                        auto tmp14 = out_ptr0[static_cast<long>(x1)];
                        auto tmp17 = in_ptr5[static_cast<long>(x1)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp7 = static_cast<float>(0.00015943877551020407);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                        auto tmp12 = decltype(tmp5)(tmp5 * tmp11);
                        auto tmp13 = decltype(tmp2)(tmp2 - tmp12);
                        auto tmp15 = decltype(tmp14)(tmp14 * tmp7);
                        auto tmp16 = decltype(tmp13)(tmp13 - tmp15);
                        auto tmp18 = decltype(tmp9)(tmp9 * tmp17);
                        auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                        out_ptr3[static_cast<long>(x2 + (784L*x1) + (15680L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_threshold_backward_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x2) + (94080L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (120L*x2) + (94080L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_67 = async_compile.cpp('''
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_68 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (60L*x2) + (47040L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(60L + x0 + (120L*x2) + (94080L*x1)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(60L + x0 + (120L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(60L + x0 + (120L*x1)));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (60L*x2) + (47040L*x1)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp5 = static_cast<float>(784.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = tmp3 + tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp15 = tmp11 * tmp14;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                            tmp_acc1_vec = tmp_acc1_vec + tmp15;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(56L); x0<static_cast<long>(60L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0 + (60L*x2) + (47040L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(60L + x0 + (120L*x2) + (94080L*x1))];
                            auto tmp2 = in_ptr2[static_cast<long>(60L + x0 + (120L*x1))];
                            auto tmp4 = in_ptr3[static_cast<long>(60L + x0 + (120L*x1))];
                            auto tmp10 = in_ptr4[static_cast<long>(x0 + (60L*x2) + (47040L*x1))];
                            auto tmp11 = in_ptr5[static_cast<long>(x0)];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp5 = static_cast<float>(784.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = decltype(tmp3)(tmp3 + tmp6);
                            auto tmp8 = static_cast<float>(0.0);
                            auto tmp9 = tmp0 ? tmp8 : tmp7;
                            auto tmp12 = decltype(tmp10)(tmp10 - tmp11);
                            auto tmp13 = decltype(tmp9)(tmp9 * tmp12);
                            tmp_acc0 = tmp_acc0 + tmp9;
                            tmp_acc1 = tmp_acc1 + tmp13;
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (60L*x1) + (47040L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(60L + x2 + (120L*x1) + (94080L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(60L + x2 + (120L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(60L + x2 + (120L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (60L*x1) + (47040L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = static_cast<float>(784.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp8 = tmp3 + tmp7;
                        auto tmp9 = static_cast<float>(0.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = decltype(tmp10)::blendv(tmp8, tmp10, tmp0);
                        auto tmp14 = tmp12 - tmp13;
                        auto tmp16 = static_cast<float>(0.00015943877551020407);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 * tmp17;
                        auto tmp20 = tmp19 * tmp19;
                        auto tmp21 = tmp18 * tmp20;
                        auto tmp22 = tmp14 * tmp21;
                        auto tmp23 = tmp11 - tmp22;
                        auto tmp25 = tmp24 * tmp17;
                        auto tmp26 = tmp23 - tmp25;
                        tmp26.store(out_ptr2 + static_cast<long>(x2 + (60L*x1) + (47040L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(56L); x2<static_cast<long>(60L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (60L*x1) + (47040L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(60L + x2 + (120L*x1) + (94080L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(60L + x2 + (120L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(60L + x2 + (120L*x0))];
                        auto tmp10 = in_ptr4[static_cast<long>(x2 + (60L*x1) + (47040L*x0))];
                        auto tmp11 = in_ptr5[static_cast<long>(x2)];
                        auto tmp13 = out_ptr1[static_cast<long>(x2)];
                        auto tmp16 = in_ptr6[static_cast<long>(x2)];
                        auto tmp21 = out_ptr0[static_cast<long>(x2)];
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp5 = static_cast<float>(784.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = decltype(tmp3)(tmp3 + tmp6);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp0 ? tmp8 : tmp7;
                        auto tmp12 = decltype(tmp10)(tmp10 - tmp11);
                        auto tmp14 = static_cast<float>(0.00015943877551020407);
                        auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                        auto tmp17 = decltype(tmp16)(tmp16 * tmp16);
                        auto tmp18 = decltype(tmp15)(tmp15 * tmp17);
                        auto tmp19 = decltype(tmp12)(tmp12 * tmp18);
                        auto tmp20 = decltype(tmp9)(tmp9 - tmp19);
                        auto tmp22 = decltype(tmp21)(tmp21 * tmp14);
                        auto tmp23 = decltype(tmp20)(tmp20 - tmp22);
                        out_ptr2[static_cast<long>(x2 + (60L*x1) + (47040L*x0))] = tmp23;
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
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(60L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    in_out_ptr0[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (60L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr1 + static_cast<long>(x1 + (60L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(60L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>(x1 + (60L*x0))];
                    auto tmp1 = in_ptr6[static_cast<long>(x1)];
                    auto tmp2 = in_ptr7[static_cast<long>(x1)];
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    in_out_ptr1[static_cast<long>(x1 + (60L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_69 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (60L*x1) + (47040L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (60L*x1) + (47040L*x0)));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = static_cast<float>(784.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 / tmp9;
                        auto tmp11 = tmp6 + tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp14 = decltype(tmp2)::blendv(tmp13, tmp2, tmp3);
                        tmp14.store(in_out_ptr0 + static_cast<long>(x2 + (60L*x1) + (47040L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(56L); x2<static_cast<long>(60L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (60L*x1) + (47040L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2 + (120L*x1) + (94080L*x0))];
                        auto tmp4 = in_ptr2[static_cast<long>(x2 + (120L*x0))];
                        auto tmp6 = in_ptr3[static_cast<long>(x2 + (120L*x0))];
                        auto tmp10 = in_out_ptr0[static_cast<long>(x2 + (60L*x1) + (47040L*x0))];
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp7 = static_cast<float>(784.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = decltype(tmp5)(tmp5 + tmp8);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = tmp2 ? tmp1 : tmp11;
                        in_out_ptr0[static_cast<long>(x2 + (60L*x1) + (47040L*x0))] = tmp12;
                    }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (60L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (60L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(56L); x0<static_cast<long>(60L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x0 + (60L*x1))];
                        auto tmp1 = in_ptr4[static_cast<long>(x0 + (60L*x1))];
                        auto tmp2 = in_ptr5[static_cast<long>(x0)];
                        auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        tmp_acc0 = tmp_acc0 + tmp0;
                        tmp_acc1 = tmp_acc1 + tmp4;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
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
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(60L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (60L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (60L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (60L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(60L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (60L*x0))];
                    auto tmp1 = in_ptr4[static_cast<long>(x1 + (60L*x0))];
                    auto tmp2 = in_ptr5[static_cast<long>(x1)];
                    auto tmp4 = out_ptr1[static_cast<long>(x1)];
                    auto tmp7 = in_ptr6[static_cast<long>(x1)];
                    auto tmp12 = out_ptr0[static_cast<long>(x1)];
                    auto tmp15 = in_ptr7[static_cast<long>(x1)];
                    auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                    auto tmp5 = static_cast<float>(0.00015943877551020407);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = decltype(tmp7)(tmp7 * tmp7);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp3)(tmp3 * tmp9);
                    auto tmp11 = decltype(tmp0)(tmp0 - tmp10);
                    auto tmp13 = decltype(tmp12)(tmp12 * tmp5);
                    auto tmp14 = decltype(tmp11)(tmp11 - tmp13);
                    auto tmp16 = decltype(tmp7)(tmp7 * tmp15);
                    auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                    in_out_ptr0[static_cast<long>(x1 + (60L*x0))] = tmp17;
                }
            }
        }
    }
}
''')


cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_70 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (40L*x2) + (31360L*x0)), static_cast<long>(40L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (31360L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (31360L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(250880L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(250880L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(250880L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr1 + static_cast<long>(x2 + (784L*x0) + (31360L*x1)), static_cast<long>(784L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr1 + static_cast<long>(x2 + (784L*x0) + (31360L*x1)), static_cast<long>(784L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x2) + (40L*x2_inner) + (31360L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (40L*x2) + (31360L*x0)), static_cast<long>(40L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (31360L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr5[static_cast<long>(x1 + x1_inner)];
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
                            tmp22.store(out_ptr6 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (31360L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_72 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(15680L + x2 + (784L*x0) + (31360L*x1)), static_cast<long>(784L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(15680L + x2 + (784L*x0) + (31360L*x1)), static_cast<long>(784L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (20L*x2) + (20L*x2_inner) + (15680L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for 
            for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(15680L + x2 + (784L*x0) + (31360L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (20L*x2) + (15680L*x1))];
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp0;
                            tmp_acc1 = tmp_acc1 + tmp4;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (20L*x2) + (15680L*x0)), static_cast<long>(20L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(15680L + x2 + (784L*x1) + (784L*x1_inner) + (31360L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr4[static_cast<long>(x1 + x1_inner)];
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
                            tmp22.store(out_ptr3 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (15680L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(16L); x1<static_cast<long>(20L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(15680L + x2 + (784L*x1) + (31360L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (20L*x2) + (15680L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp4 = out_ptr1[static_cast<long>(x1)];
                        auto tmp7 = in_ptr3[static_cast<long>(x1)];
                        auto tmp12 = out_ptr0[static_cast<long>(x1)];
                        auto tmp15 = in_ptr4[static_cast<long>(x1)];
                        auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                        auto tmp5 = static_cast<float>(0.00015943877551020407);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp7)(tmp7 * tmp7);
                        auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                        auto tmp10 = decltype(tmp3)(tmp3 * tmp9);
                        auto tmp11 = decltype(tmp0)(tmp0 - tmp10);
                        auto tmp13 = decltype(tmp12)(tmp12 * tmp5);
                        auto tmp14 = decltype(tmp11)(tmp11 - tmp13);
                        auto tmp16 = decltype(tmp7)(tmp7 * tmp15);
                        auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                        out_ptr3[static_cast<long>(x2 + (784L*x1) + (15680L*x0))] = tmp17;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (784L*x0) + (31360L*x1)), static_cast<long>(784L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (784L*x0) + (31360L*x1)), static_cast<long>(784L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (20L*x2) + (20L*x2_inner) + (15680L*x1)));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (20L*x2) + (20L*x2_inner) + (15680L*x1)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp4 - tmp5;
                                auto tmp7 = tmp3 * tmp6;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp7;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for 
            for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (784L*x0) + (31360L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (20L*x2) + (15680L*x1))];
                            auto tmp3 = in_ptr2[static_cast<long>(x0 + (20L*x2) + (15680L*x1))];
                            auto tmp4 = in_ptr3[static_cast<long>(x0)];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                            auto tmp6 = decltype(tmp2)(tmp2 * tmp5);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp6;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (20L*x2) + (15680L*x0)), static_cast<long>(20L), tmp1, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (20L*x2) + (15680L*x0)), static_cast<long>(20L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (31360L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp12 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp18 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                            auto tmp22 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(0.00015943877551020407);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp13 = decltype(tmp12)(tmp12 * tmp12);
                            auto tmp14 = decltype(tmp11)(tmp11 * tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            auto tmp17 = tmp3 - tmp16;
                            auto tmp19 = decltype(tmp18)(tmp18 * tmp10);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp17 - tmp20;
                            auto tmp23 = decltype(tmp12)(tmp12 * tmp22);
                            auto tmp24 = at::vec::Vectorized<float>(tmp23);
                            auto tmp25 = tmp21 * tmp24;
                            tmp25.store(out_ptr3 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (15680L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(16L); x1<static_cast<long>(20L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (784L*x1) + (31360L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (20L*x2) + (15680L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x1 + (20L*x2) + (15680L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp6 = out_ptr1[static_cast<long>(x1)];
                        auto tmp9 = in_ptr4[static_cast<long>(x1)];
                        auto tmp14 = out_ptr0[static_cast<long>(x1)];
                        auto tmp17 = in_ptr5[static_cast<long>(x1)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp7 = static_cast<float>(0.00015943877551020407);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                        auto tmp12 = decltype(tmp5)(tmp5 * tmp11);
                        auto tmp13 = decltype(tmp2)(tmp2 - tmp12);
                        auto tmp15 = decltype(tmp14)(tmp14 * tmp7);
                        auto tmp16 = decltype(tmp13)(tmp13 - tmp15);
                        auto tmp18 = decltype(tmp9)(tmp9 * tmp17);
                        auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                        out_ptr3[static_cast<long>(x2 + (784L*x1) + (15680L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_threshold_backward_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x2) + (56448L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (72L*x2) + (56448L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
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
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mul_native_batch_norm_backward_76 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (72L*x2) + (56448L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (72L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (72L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (72L*x2) + (56448L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = static_cast<float>(784.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = tmp2 + tmp6;
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(72L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (72L*x1) + (56448L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (72L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (72L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (72L*x1) + (56448L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = static_cast<float>(784.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 + tmp6;
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
                        tmp25.store(in_out_ptr0 + static_cast<long>(x2 + (72L*x1) + (56448L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_77 = async_compile.cpp('''
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (36L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(36L + x0 + (72L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (36L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (36L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(36L + x0 + (72L*x1))];
                        auto tmp4 = in_ptr2[static_cast<long>(x0 + (36L*x1))];
                        auto tmp5 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = tmp0 ? tmp2 : tmp1;
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp3;
                        tmp_acc1 = tmp_acc1 + tmp7;
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (36L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(36L + x1 + (72L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (36L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (36L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(32L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (36L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(36L + x1 + (72L*x0))];
                    auto tmp4 = in_ptr2[static_cast<long>(x1 + (36L*x0))];
                    auto tmp5 = in_ptr3[static_cast<long>(x1)];
                    auto tmp7 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr4[static_cast<long>(x1)];
                    auto tmp15 = out_ptr0[static_cast<long>(x1)];
                    auto tmp18 = in_ptr5[static_cast<long>(x1)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = tmp0 ? tmp2 : tmp1;
                    auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                    auto tmp8 = static_cast<float>(3.985969387755102e-05);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp11 = decltype(tmp10)(tmp10 * tmp10);
                    auto tmp12 = decltype(tmp9)(tmp9 * tmp11);
                    auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                    auto tmp14 = decltype(tmp3)(tmp3 - tmp13);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                    auto tmp17 = decltype(tmp14)(tmp14 - tmp16);
                    auto tmp19 = decltype(tmp10)(tmp10 * tmp18);
                    auto tmp20 = decltype(tmp17)(tmp17 * tmp19);
                    out_ptr3[static_cast<long>(x1 + (36L*x0))] = tmp20;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_78 = async_compile.cpp('''
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
    auto in_ptr2 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (36L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (72L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (36L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (36L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (36L*x1))];
                        auto tmp3 = in_ptr1[static_cast<long>(x0 + (72L*x1))];
                        auto tmp4 = in_ptr2[static_cast<long>(x0 + (36L*x1))];
                        auto tmp7 = in_ptr3[static_cast<long>(x0 + (36L*x1))];
                        auto tmp8 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp6 = tmp2 ? tmp1 : tmp5;
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        tmp_acc0 = tmp_acc0 + tmp6;
                        tmp_acc1 = tmp_acc1 + tmp10;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (36L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (36L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (36L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (36L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(32L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (36L*x0))];
                    auto tmp3 = in_ptr1[static_cast<long>(x1 + (72L*x0))];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x1 + (36L*x0))];
                    auto tmp7 = in_ptr3[static_cast<long>(x1 + (36L*x0))];
                    auto tmp8 = in_ptr4[static_cast<long>(x1)];
                    auto tmp10 = out_ptr1[static_cast<long>(x1)];
                    auto tmp13 = in_ptr5[static_cast<long>(x1)];
                    auto tmp18 = out_ptr0[static_cast<long>(x1)];
                    auto tmp21 = in_ptr6[static_cast<long>(x1)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = tmp2 ? tmp1 : tmp5;
                    auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                    auto tmp11 = static_cast<float>(3.985969387755102e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp13)(tmp13 * tmp13);
                    auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                    auto tmp16 = decltype(tmp9)(tmp9 * tmp15);
                    auto tmp17 = decltype(tmp6)(tmp6 - tmp16);
                    auto tmp19 = decltype(tmp18)(tmp18 * tmp11);
                    auto tmp20 = decltype(tmp17)(tmp17 - tmp19);
                    auto tmp22 = decltype(tmp13)(tmp13 * tmp21);
                    auto tmp23 = decltype(tmp20)(tmp20 * tmp22);
                    in_out_ptr0[static_cast<long>(x1 + (36L*x0))] = tmp23;
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    in_out_ptr1[static_cast<long>(x0)] = tmp2;
                }
            }
        }
    }
}
''')


cpp_fused_add_as_strided_scatter_convolution_backward_copy_native_batch_norm_backward_79 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (24L*x1) + (24L*x1_inner) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (24L*x1) + (24L*x1_inner) + (75264L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr0 + static_cast<long>(x1 + (3136L*x2) + (75264L*x0)), static_cast<long>(3136L));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(37632L + x2 + (3136L*x0) + (75264L*x1)), static_cast<long>(3136L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(37632L + x2 + (3136L*x0) + (75264L*x1)), static_cast<long>(3136L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (37632L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
            #pragma omp for 
            for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr2[static_cast<long>(37632L + x2 + (3136L*x0) + (75264L*x1))];
                            auto tmp1 = in_ptr2[static_cast<long>(x0 + (12L*x2) + (37632L*x1))];
                            auto tmp2 = in_ptr3[static_cast<long>(x0)];
                            auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp0;
                            tmp_acc1 = tmp_acc1 + tmp4;
                        }
                    }
                    out_ptr3[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr4[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr4[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr5[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (12L*x2) + (37632L*x0)), static_cast<long>(12L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(37632L + x2 + (3136L*x1) + (3136L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(3.985969387755102e-05);
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
                            tmp22.store(out_ptr6 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (37632L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(37632L + x2 + (3136L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr2[static_cast<long>(x1 + (12L*x2) + (37632L*x0))];
                        auto tmp2 = in_ptr3[static_cast<long>(x1)];
                        auto tmp4 = out_ptr4[static_cast<long>(x1)];
                        auto tmp7 = in_ptr4[static_cast<long>(x1)];
                        auto tmp12 = out_ptr3[static_cast<long>(x1)];
                        auto tmp15 = in_ptr5[static_cast<long>(x1)];
                        auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                        auto tmp5 = static_cast<float>(3.985969387755102e-05);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp7)(tmp7 * tmp7);
                        auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                        auto tmp10 = decltype(tmp3)(tmp3 * tmp9);
                        auto tmp11 = decltype(tmp0)(tmp0 - tmp10);
                        auto tmp13 = decltype(tmp12)(tmp12 * tmp5);
                        auto tmp14 = decltype(tmp11)(tmp11 - tmp13);
                        auto tmp16 = decltype(tmp7)(tmp7 * tmp15);
                        auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                        out_ptr6[static_cast<long>(x2 + (3136L*x1) + (37632L*x0))] = tmp17;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_80 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (3136L*x0) + (75264L*x1)), static_cast<long>(3136L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (3136L*x0) + (75264L*x1)), static_cast<long>(3136L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (37632L*x1)));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (37632L*x1)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp4 - tmp5;
                                auto tmp7 = tmp3 * tmp6;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp7;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for 
            for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (3136L*x0) + (75264L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (12L*x2) + (37632L*x1))];
                            auto tmp3 = in_ptr2[static_cast<long>(x0 + (12L*x2) + (37632L*x1))];
                            auto tmp4 = in_ptr3[static_cast<long>(x0)];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                            auto tmp6 = decltype(tmp2)(tmp2 * tmp5);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp6;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (12L*x2) + (37632L*x0)), static_cast<long>(12L), tmp1, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (12L*x2) + (37632L*x0)), static_cast<long>(12L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp12 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp18 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                            auto tmp22 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(3.985969387755102e-05);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp13 = decltype(tmp12)(tmp12 * tmp12);
                            auto tmp14 = decltype(tmp11)(tmp11 * tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            auto tmp17 = tmp3 - tmp16;
                            auto tmp19 = decltype(tmp18)(tmp18 * tmp10);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp17 - tmp20;
                            auto tmp23 = decltype(tmp12)(tmp12 * tmp22);
                            auto tmp24 = at::vec::Vectorized<float>(tmp23);
                            auto tmp25 = tmp21 * tmp24;
                            tmp25.store(out_ptr3 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (37632L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (3136L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x2) + (37632L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x1 + (12L*x2) + (37632L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp6 = out_ptr1[static_cast<long>(x1)];
                        auto tmp9 = in_ptr4[static_cast<long>(x1)];
                        auto tmp14 = out_ptr0[static_cast<long>(x1)];
                        auto tmp17 = in_ptr5[static_cast<long>(x1)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp7 = static_cast<float>(3.985969387755102e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                        auto tmp12 = decltype(tmp5)(tmp5 * tmp11);
                        auto tmp13 = decltype(tmp2)(tmp2 - tmp12);
                        auto tmp15 = decltype(tmp14)(tmp14 * tmp7);
                        auto tmp16 = decltype(tmp13)(tmp13 - tmp15);
                        auto tmp18 = decltype(tmp9)(tmp9 * tmp17);
                        auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                        out_ptr3[static_cast<long>(x2 + (3136L*x1) + (37632L*x0))] = tmp19;
                    }
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (36L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(36L + x0 + (72L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (36L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (36L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(36L + x0 + (72L*x1))];
                        auto tmp4 = in_ptr2[static_cast<long>(x0 + (36L*x1))];
                        auto tmp5 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = tmp0 ? tmp2 : tmp1;
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp3;
                        tmp_acc1 = tmp_acc1 + tmp7;
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (36L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(36L + x1 + (72L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (36L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (36L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(32L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (36L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(36L + x1 + (72L*x0))];
                    auto tmp4 = in_ptr2[static_cast<long>(x1 + (36L*x0))];
                    auto tmp5 = in_ptr3[static_cast<long>(x1)];
                    auto tmp7 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr4[static_cast<long>(x1)];
                    auto tmp15 = out_ptr0[static_cast<long>(x1)];
                    auto tmp18 = in_ptr5[static_cast<long>(x1)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = tmp0 ? tmp2 : tmp1;
                    auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                    auto tmp8 = static_cast<float>(3.985969387755102e-05);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp11 = decltype(tmp10)(tmp10 * tmp10);
                    auto tmp12 = decltype(tmp9)(tmp9 * tmp11);
                    auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                    auto tmp14 = decltype(tmp3)(tmp3 - tmp13);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                    auto tmp17 = decltype(tmp14)(tmp14 - tmp16);
                    auto tmp19 = decltype(tmp10)(tmp10 * tmp18);
                    auto tmp20 = decltype(tmp17)(tmp17 * tmp19);
                    out_ptr3[static_cast<long>(x1 + (36L*x0))] = tmp20;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_82 = async_compile.cpp('''
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
    auto in_ptr2 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (36L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (72L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (36L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (36L*x1)));
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
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (36L*x1))];
                        auto tmp3 = in_ptr1[static_cast<long>(x0 + (72L*x1))];
                        auto tmp4 = in_ptr2[static_cast<long>(x0 + (36L*x1))];
                        auto tmp7 = in_ptr3[static_cast<long>(x0 + (36L*x1))];
                        auto tmp8 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = tmp0 <= tmp1;
                        auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                        auto tmp6 = tmp2 ? tmp1 : tmp5;
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = decltype(tmp6)(tmp6 * tmp9);
                        tmp_acc0 = tmp_acc0 + tmp6;
                        tmp_acc1 = tmp_acc1 + tmp10;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (36L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (36L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (36L*x0)));
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (36L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(32L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (36L*x0))];
                    auto tmp3 = in_ptr1[static_cast<long>(x1 + (72L*x0))];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x1 + (36L*x0))];
                    auto tmp7 = in_ptr3[static_cast<long>(x1 + (36L*x0))];
                    auto tmp8 = in_ptr4[static_cast<long>(x1)];
                    auto tmp10 = out_ptr1[static_cast<long>(x1)];
                    auto tmp13 = in_ptr5[static_cast<long>(x1)];
                    auto tmp18 = out_ptr0[static_cast<long>(x1)];
                    auto tmp21 = in_ptr6[static_cast<long>(x1)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = tmp2 ? tmp1 : tmp5;
                    auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                    auto tmp11 = static_cast<float>(3.985969387755102e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp13)(tmp13 * tmp13);
                    auto tmp15 = decltype(tmp12)(tmp12 * tmp14);
                    auto tmp16 = decltype(tmp9)(tmp9 * tmp15);
                    auto tmp17 = decltype(tmp6)(tmp6 - tmp16);
                    auto tmp19 = decltype(tmp18)(tmp18 * tmp11);
                    auto tmp20 = decltype(tmp17)(tmp17 - tmp19);
                    auto tmp22 = decltype(tmp13)(tmp13 * tmp21);
                    auto tmp23 = decltype(tmp20)(tmp20 * tmp22);
                    in_out_ptr0[static_cast<long>(x1 + (36L*x0))] = tmp23;
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    in_out_ptr1[static_cast<long>(x0)] = tmp2;
                }
            }
        }
    }
}
''')


cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_83 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (24L*x2) + (75264L*x0)), static_cast<long>(24L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr1 + static_cast<long>(x2 + (3136L*x0) + (75264L*x1)), static_cast<long>(3136L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr1 + static_cast<long>(x2 + (3136L*x0) + (75264L*x1)), static_cast<long>(3136L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (24L*x2) + (24L*x2_inner) + (75264L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (24L*x2) + (75264L*x0)), static_cast<long>(24L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(3.985969387755102e-05);
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
                            tmp22.store(out_ptr6 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_84 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_85 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(37632L + x2 + (3136L*x0) + (75264L*x1)), static_cast<long>(3136L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(37632L + x2 + (3136L*x0) + (75264L*x1)), static_cast<long>(3136L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (37632L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for 
            for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(37632L + x2 + (3136L*x0) + (75264L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (12L*x2) + (37632L*x1))];
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                            tmp_acc0 = tmp_acc0 + tmp0;
                            tmp_acc1 = tmp_acc1 + tmp4;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (12L*x2) + (37632L*x0)), static_cast<long>(12L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(37632L + x2 + (3136L*x1) + (3136L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(3.985969387755102e-05);
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
                            tmp22.store(out_ptr3 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (37632L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(37632L + x2 + (3136L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x2) + (37632L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x1)];
                        auto tmp4 = out_ptr1[static_cast<long>(x1)];
                        auto tmp7 = in_ptr3[static_cast<long>(x1)];
                        auto tmp12 = out_ptr0[static_cast<long>(x1)];
                        auto tmp15 = in_ptr4[static_cast<long>(x1)];
                        auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                        auto tmp5 = static_cast<float>(3.985969387755102e-05);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp7)(tmp7 * tmp7);
                        auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                        auto tmp10 = decltype(tmp3)(tmp3 * tmp9);
                        auto tmp11 = decltype(tmp0)(tmp0 - tmp10);
                        auto tmp13 = decltype(tmp12)(tmp12 * tmp5);
                        auto tmp14 = decltype(tmp11)(tmp11 - tmp13);
                        auto tmp16 = decltype(tmp7)(tmp7 * tmp15);
                        auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                        out_ptr3[static_cast<long>(x2 + (3136L*x1) + (37632L*x0))] = tmp17;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_86 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (3136L*x0) + (75264L*x1)), static_cast<long>(3136L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (3136L*x0) + (75264L*x1)), static_cast<long>(3136L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (37632L*x1)));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (37632L*x1)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp4 - tmp5;
                                auto tmp7 = tmp3 * tmp6;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp7;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for 
            for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (3136L*x0) + (75264L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (12L*x2) + (37632L*x1))];
                            auto tmp3 = in_ptr2[static_cast<long>(x0 + (12L*x2) + (37632L*x1))];
                            auto tmp4 = in_ptr3[static_cast<long>(x0)];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                            auto tmp6 = decltype(tmp2)(tmp2 * tmp5);
                            tmp_acc0 = tmp_acc0 + tmp2;
                            tmp_acc1 = tmp_acc1 + tmp6;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (12L*x2) + (37632L*x0)), static_cast<long>(12L), tmp1, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (12L*x2) + (37632L*x0)), static_cast<long>(12L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (75264L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp12 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp18 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                            auto tmp22 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(3.985969387755102e-05);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp13 = decltype(tmp12)(tmp12 * tmp12);
                            auto tmp14 = decltype(tmp11)(tmp11 * tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            auto tmp17 = tmp3 - tmp16;
                            auto tmp19 = decltype(tmp18)(tmp18 * tmp10);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp17 - tmp20;
                            auto tmp23 = decltype(tmp12)(tmp12 * tmp22);
                            auto tmp24 = at::vec::Vectorized<float>(tmp23);
                            auto tmp25 = tmp21 * tmp24;
                            tmp25.store(out_ptr3 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (37632L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (3136L*x1) + (75264L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x2) + (37632L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x1 + (12L*x2) + (37632L*x0))];
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp6 = out_ptr1[static_cast<long>(x1)];
                        auto tmp9 = in_ptr4[static_cast<long>(x1)];
                        auto tmp14 = out_ptr0[static_cast<long>(x1)];
                        auto tmp17 = in_ptr5[static_cast<long>(x1)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp7 = static_cast<float>(3.985969387755102e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                        auto tmp12 = decltype(tmp5)(tmp5 * tmp11);
                        auto tmp13 = decltype(tmp2)(tmp2 - tmp12);
                        auto tmp15 = decltype(tmp14)(tmp14 * tmp7);
                        auto tmp16 = decltype(tmp13)(tmp13 - tmp15);
                        auto tmp18 = decltype(tmp9)(tmp9 * tmp17);
                        auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                        out_ptr3[static_cast<long>(x2 + (3136L*x1) + (37632L*x0))] = tmp19;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_87 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (48L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (48L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (48L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_88 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(24L + x0 + (48L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (24L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(24L + x1 + (48L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_89 = async_compile.cpp('''
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
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (48L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (24L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
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
                    auto tmp12 = static_cast<float>(9.964923469387754e-06);
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
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


cpp_fused_add_as_strided_scatter_convolution_backward_copy_native_batch_norm_backward_90 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (200704L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (200704L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr0 + static_cast<long>(x1 + (12544L*x2) + (200704L*x0)), static_cast<long>(12544L));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(12544L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(100352L + x2 + (12544L*x0) + (200704L*x1)), static_cast<long>(12544L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(100352L + x2 + (12544L*x0) + (200704L*x1)), static_cast<long>(12544L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (8L*x2) + (8L*x2_inner) + (100352L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp4 = tmp2 - tmp3;
                                auto tmp5 = tmp1 * tmp4;
                                tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                tmp_acc1_vec = tmp_acc1_vec + tmp5;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12544L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (8L*x2) + (100352L*x0)), static_cast<long>(8L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(100352L + x2 + (12544L*x1) + (12544L*x1_inner) + (200704L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = out_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(9.964923469387754e-06);
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
                            tmp22.store(out_ptr6 + static_cast<long>(x2 + (12544L*x1) + (12544L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_91 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(12544L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (12544L*x0) + (200704L*x1)), static_cast<long>(12544L), tmp0, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x2 + (12544L*x0) + (200704L*x1)), static_cast<long>(12544L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (8L*x2) + (8L*x2_inner) + (100352L*x1)));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (8L*x2) + (8L*x2_inner) + (100352L*x1)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp6 = tmp4 - tmp5;
                                auto tmp7 = tmp3 * tmp6;
                                tmp_acc0_vec = tmp_acc0_vec + tmp3;
                                tmp_acc1_vec = tmp_acc1_vec + tmp7;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12544L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (8L*x2) + (100352L*x0)), static_cast<long>(8L), tmp1, 8);
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x1 + (8L*x2) + (100352L*x0)), static_cast<long>(8L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (12544L*x1) + (12544L*x1_inner) + (200704L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp12 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp18 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                            auto tmp22 = in_ptr5[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(9.964923469387754e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp13 = decltype(tmp12)(tmp12 * tmp12);
                            auto tmp14 = decltype(tmp11)(tmp11 * tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            auto tmp17 = tmp3 - tmp16;
                            auto tmp19 = decltype(tmp18)(tmp18 * tmp10);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp17 - tmp20;
                            auto tmp23 = decltype(tmp12)(tmp12 * tmp22);
                            auto tmp24 = at::vec::Vectorized<float>(tmp23);
                            auto tmp25 = tmp21 * tmp24;
                            tmp25.store(out_ptr3 + static_cast<long>(x2 + (12544L*x1) + (12544L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_92 = async_compile.cpp('''
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (8L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(8L + x0 + (16L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (8L*x1)));
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
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (8L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(8L + x1 + (16L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (8L*x0)));
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
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (8L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_93 = async_compile.cpp('''
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
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (8L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (8L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (8L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (8L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (8L*x0)));
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
                    auto tmp12 = static_cast<float>(9.964923469387754e-06);
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
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (8L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_batch_norm_backward_threshold_backward_94 = async_compile.cpp('''
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
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(12544L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (12544L*x0) + (200704L*x1)), static_cast<long>(12544L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (12544L*x0) + (200704L*x1)), static_cast<long>(12544L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16L*x2) + (16L*x2_inner) + (200704L*x1)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x2) + (16L*x2_inner) + (200704L*x1)));
                                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (16L*x2) + (16L*x2_inner) + (200704L*x1)));
                                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                                auto tmp1 = static_cast<float>(0.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                                auto tmp7 = tmp5 + tmp6;
                                auto tmp8 = decltype(tmp2)::blendv(tmp7, tmp2, tmp3);
                                auto tmp11 = tmp9 - tmp10;
                                auto tmp12 = tmp8 * tmp11;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                                tmp_acc1_vec = tmp_acc1_vec + tmp12;
                            }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp4[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (12544L*x2) + (200704L*x0)), static_cast<long>(12544L), tmp4, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (200704L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (200704L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (200704L*x0)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp7 = tmp5 + tmp6;
                            auto tmp8 = decltype(tmp2)::blendv(tmp7, tmp2, tmp3);
                            auto tmp11 = tmp9 - tmp10;
                            auto tmp13 = static_cast<float>(9.964923469387754e-06);
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
                            tmp26.store(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (200704L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_15, primals_17, primals_18, primals_20, primals_21, primals_23, primals_24, primals_26, primals_27, primals_29, primals_30, primals_32, primals_33, primals_35, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_47, primals_48, primals_50, primals_51, primals_53, primals_54, primals_56, primals_57, primals_59, primals_60, primals_62, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_159, primals_161, primals_162, primals_164, primals_165, primals_167, primals_168, primals_170, primals_171, primals_173, primals_174, primals_176, primals_177, primals_179, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_234, primals_236, primals_237, primals_239, primals_240, primals_242, primals_243, primals_245, primals_246, primals_248, primals_249, primals_251, primals_252, primals_254, primals_255, primals_257, primals_258, primals_260, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_513, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, slice_3, convolution_3, squeeze_10, add_19, convolution_4, squeeze_13, slice_11, convolution_5, squeeze_16, relu_3, convolution_6, squeeze_19, slice_14, convolution_7, squeeze_22, add_40, convolution_8, squeeze_25, add_45, convolution_9, squeeze_28, convolution_10, squeeze_31, add_55, convolution_11, squeeze_34, slice_22, convolution_12, squeeze_37, relu_5, convolution_13, squeeze_40, slice_25, convolution_14, squeeze_43, add_76, convolution_15, squeeze_46, slice_33, convolution_16, squeeze_49, relu_7, convolution_17, squeeze_52, slice_36, convolution_18, squeeze_55, add_97, mean, relu_9, div, mul_133, convolution_21, squeeze_58, add_103, convolution_22, squeeze_61, convolution_23, squeeze_64, add_113, convolution_24, squeeze_67, slice_44, convolution_25, squeeze_70, relu_10, convolution_26, squeeze_73, cat_8, mean_1, relu_12, div_1, mul_176, convolution_29, squeeze_76, add_135, convolution_30, squeeze_79, slice_55, convolution_31, squeeze_82, relu_13, convolution_32, squeeze_85, slice_58, convolution_33, squeeze_88, add_156, convolution_34, squeeze_91, add_161, convolution_35, squeeze_94, convolution_36, squeeze_97, add_171, convolution_37, squeeze_100, slice_66, convolution_38, squeeze_103, relu_15, convolution_39, squeeze_106, slice_69, convolution_40, squeeze_109, add_192, convolution_41, squeeze_112, slice_77, convolution_42, squeeze_115, relu_17, convolution_43, squeeze_118, slice_80, convolution_44, squeeze_121, add_213, convolution_45, squeeze_124, slice_88, convolution_46, squeeze_127, relu_19, convolution_47, squeeze_130, slice_91, convolution_48, squeeze_133, add_234, convolution_49, squeeze_136, slice_99, convolution_50, squeeze_139, relu_21, convolution_51, squeeze_142, cat_18, mean_2, relu_23, div_2, mul_338, convolution_54, squeeze_145, add_256, convolution_55, squeeze_148, convolution_56, squeeze_151, add_266, convolution_57, squeeze_154, slice_110, convolution_58, squeeze_157, relu_24, convolution_59, squeeze_160, cat_20, mean_3, relu_26, div_3, mul_381, convolution_62, squeeze_163, add_288, convolution_63, squeeze_166, slice_121, convolution_64, squeeze_169, relu_27, convolution_65, squeeze_172, slice_124, convolution_66, squeeze_175, add_309, mean_4, relu_29, div_4, mul_417, convolution_69, squeeze_178, add_315, convolution_70, squeeze_181, convolution_71, squeeze_184, add_325, convolution_72, squeeze_187, slice_132, convolution_73, squeeze_190, relu_30, convolution_74, squeeze_193, slice_135, convolution_75, squeeze_196, add_346, convolution_76, squeeze_199, slice_143, convolution_77, squeeze_202, relu_32, convolution_78, squeeze_205, cat_26, mean_5, relu_34, div_5, mul_488, convolution_81, squeeze_208, add_368, convolution_82, squeeze_211, slice_154, convolution_83, squeeze_214, relu_35, convolution_84, squeeze_217, slice_157, convolution_85, squeeze_220, add_389, convolution_86, squeeze_223, slice_165, convolution_87, squeeze_226, relu_37, convolution_88, squeeze_229, cat_30, mean_6, relu_39, div_6, mul_545, convolution_91, squeeze_232, add_411, convolution_92, squeeze_235, slice_176, convolution_93, squeeze_238, mean_7, view_1, permute_1, le, le_1, unsqueeze_322, unsqueeze_334, unsqueeze_346, bitwise_and, le_3, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, le_5, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, bitwise_and_1, le_8, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, le_10, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, bitwise_and_2, unsqueeze_574, le_13, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, bitwise_and_3, le_16, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, bitwise_and_4, le_19, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, le_21, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, le_23, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, le_25, unsqueeze_850, unsqueeze_862, unsqueeze_874, unsqueeze_886, unsqueeze_898, unsqueeze_910, unsqueeze_922, le_27, unsqueeze_934, unsqueeze_946, unsqueeze_958, unsqueeze_970, bitwise_and_5, le_30, unsqueeze_982, unsqueeze_994, unsqueeze_1006, unsqueeze_1018, unsqueeze_1030, unsqueeze_1042, bitwise_and_6, unsqueeze_1054, le_33, unsqueeze_1066, unsqueeze_1078, unsqueeze_1090, unsqueeze_1102, le_35, unsqueeze_1114, unsqueeze_1126, unsqueeze_1138, unsqueeze_1150, unsqueeze_1162, unsqueeze_1174, unsqueeze_1186, le_37, unsqueeze_1198, unsqueeze_1210, unsqueeze_1222, unsqueeze_1234, le_39, unsqueeze_1246, unsqueeze_1258, unsqueeze_1270, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (960, ), (1, ))
    assert_size_stride(primals_5, (16, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_8, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_9, (8, ), (1, ))
    assert_size_stride(primals_11, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_12, (8, ), (1, ))
    assert_size_stride(primals_14, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_15, (8, ), (1, ))
    assert_size_stride(primals_17, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_18, (8, ), (1, ))
    assert_size_stride(primals_20, (24, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_21, (24, ), (1, ))
    assert_size_stride(primals_23, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_24, (24, ), (1, ))
    assert_size_stride(primals_26, (48, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_27, (48, ), (1, ))
    assert_size_stride(primals_29, (12, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_30, (12, ), (1, ))
    assert_size_stride(primals_32, (12, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_33, (12, ), (1, ))
    assert_size_stride(primals_35, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_36, (16, ), (1, ))
    assert_size_stride(primals_38, (24, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_39, (24, ), (1, ))
    assert_size_stride(primals_41, (36, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_42, (36, ), (1, ))
    assert_size_stride(primals_44, (36, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_45, (36, ), (1, ))
    assert_size_stride(primals_47, (12, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_48, (12, ), (1, ))
    assert_size_stride(primals_50, (12, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_51, (12, ), (1, ))
    assert_size_stride(primals_53, (36, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_54, (36, ), (1, ))
    assert_size_stride(primals_56, (36, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_57, (36, ), (1, ))
    assert_size_stride(primals_59, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_60, (72, ), (1, ))
    assert_size_stride(primals_62, (20, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_64, (72, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_66, (20, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_67, (20, ), (1, ))
    assert_size_stride(primals_69, (20, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_70, (20, ), (1, ))
    assert_size_stride(primals_72, (24, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_73, (24, ), (1, ))
    assert_size_stride(primals_75, (40, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_76, (40, ), (1, ))
    assert_size_stride(primals_78, (60, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_79, (60, ), (1, ))
    assert_size_stride(primals_81, (60, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_82, (60, ), (1, ))
    assert_size_stride(primals_84, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_86, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_88, (20, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_89, (20, ), (1, ))
    assert_size_stride(primals_91, (20, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_92, (20, ), (1, ))
    assert_size_stride(primals_94, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_95, (120, ), (1, ))
    assert_size_stride(primals_97, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_98, (120, ), (1, ))
    assert_size_stride(primals_100, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_101, (240, ), (1, ))
    assert_size_stride(primals_103, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_104, (40, ), (1, ))
    assert_size_stride(primals_106, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_107, (40, ), (1, ))
    assert_size_stride(primals_109, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_110, (40, ), (1, ))
    assert_size_stride(primals_112, (80, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_113, (80, ), (1, ))
    assert_size_stride(primals_115, (100, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_116, (100, ), (1, ))
    assert_size_stride(primals_118, (100, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_119, (100, ), (1, ))
    assert_size_stride(primals_121, (40, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_122, (40, ), (1, ))
    assert_size_stride(primals_124, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_125, (40, ), (1, ))
    assert_size_stride(primals_127, (92, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_128, (92, ), (1, ))
    assert_size_stride(primals_130, (92, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_131, (92, ), (1, ))
    assert_size_stride(primals_133, (40, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_134, (40, ), (1, ))
    assert_size_stride(primals_136, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_137, (40, ), (1, ))
    assert_size_stride(primals_139, (92, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_140, (92, ), (1, ))
    assert_size_stride(primals_142, (92, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_143, (92, ), (1, ))
    assert_size_stride(primals_145, (40, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_146, (40, ), (1, ))
    assert_size_stride(primals_148, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_149, (40, ), (1, ))
    assert_size_stride(primals_151, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_152, (240, ), (1, ))
    assert_size_stride(primals_154, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_155, (240, ), (1, ))
    assert_size_stride(primals_157, (120, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_159, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_161, (56, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_162, (56, ), (1, ))
    assert_size_stride(primals_164, (56, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_165, (56, ), (1, ))
    assert_size_stride(primals_167, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_168, (80, ), (1, ))
    assert_size_stride(primals_170, (112, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_171, (112, ), (1, ))
    assert_size_stride(primals_173, (336, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_174, (336, ), (1, ))
    assert_size_stride(primals_176, (336, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_177, (336, ), (1, ))
    assert_size_stride(primals_179, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_181, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_183, (56, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_184, (56, ), (1, ))
    assert_size_stride(primals_186, (56, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_187, (56, ), (1, ))
    assert_size_stride(primals_189, (336, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_190, (336, ), (1, ))
    assert_size_stride(primals_192, (336, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_193, (336, ), (1, ))
    assert_size_stride(primals_195, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_196, (672, ), (1, ))
    assert_size_stride(primals_198, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_200, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_202, (80, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_203, (80, ), (1, ))
    assert_size_stride(primals_205, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_206, (80, ), (1, ))
    assert_size_stride(primals_208, (112, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_209, (112, ), (1, ))
    assert_size_stride(primals_211, (160, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_212, (160, ), (1, ))
    assert_size_stride(primals_214, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_215, (480, ), (1, ))
    assert_size_stride(primals_217, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_218, (480, ), (1, ))
    assert_size_stride(primals_220, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_221, (80, ), (1, ))
    assert_size_stride(primals_223, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_224, (80, ), (1, ))
    assert_size_stride(primals_226, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_227, (480, ), (1, ))
    assert_size_stride(primals_229, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_230, (480, ), (1, ))
    assert_size_stride(primals_232, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_234, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_236, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_237, (80, ), (1, ))
    assert_size_stride(primals_239, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_240, (80, ), (1, ))
    assert_size_stride(primals_242, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_243, (480, ), (1, ))
    assert_size_stride(primals_245, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_246, (480, ), (1, ))
    assert_size_stride(primals_248, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_249, (80, ), (1, ))
    assert_size_stride(primals_251, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_252, (80, ), (1, ))
    assert_size_stride(primals_254, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_255, (480, ), (1, ))
    assert_size_stride(primals_257, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_258, (480, ), (1, ))
    assert_size_stride(primals_260, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_262, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_264, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_265, (80, ), (1, ))
    assert_size_stride(primals_267, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_268, (80, ), (1, ))
    assert_size_stride(primals_270, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_271, (1280, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_513, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_1, (16, ), (1, ))
    assert_size_stride(relu, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_1, (8, 8, 112, 112), (100352, 1, 896, 8))
    assert_size_stride(squeeze_4, (8, ), (1, ))
    assert_size_stride(relu_1, (8, 8, 112, 112), (100352, 1, 896, 8))
    assert_size_stride(convolution_2, (8, 8, 112, 112), (100352, 1, 896, 8))
    assert_size_stride(squeeze_7, (8, ), (1, ))
    assert_size_stride(slice_3, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_3, (8, 8, 112, 112), (100352, 1, 896, 8))
    assert_size_stride(squeeze_10, (8, ), (1, ))
    assert_size_stride(add_19, (8, 8, 112, 112), (100352, 1, 896, 8))
    assert_size_stride(convolution_4, (8, 8, 112, 112), (100352, 1, 896, 8))
    assert_size_stride(squeeze_13, (8, ), (1, ))
    assert_size_stride(slice_11, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_5, (8, 24, 112, 112), (301056, 1, 2688, 24))
    assert_size_stride(squeeze_16, (24, ), (1, ))
    assert_size_stride(relu_3, (8, 24, 112, 112), (301056, 1, 2688, 24))
    assert_size_stride(convolution_6, (8, 24, 112, 112), (301056, 1, 2688, 24))
    assert_size_stride(squeeze_19, (24, ), (1, ))
    assert_size_stride(slice_14, (8, 48, 112, 112), (602112, 1, 5376, 48))
    assert_size_stride(convolution_7, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(squeeze_22, (48, ), (1, ))
    assert_size_stride(add_40, (8, 48, 56, 56), (150528, 1, 2688, 48))
    assert_size_stride(convolution_8, (8, 12, 56, 56), (37632, 1, 672, 12))
    assert_size_stride(squeeze_25, (12, ), (1, ))
    assert_size_stride(add_45, (8, 12, 56, 56), (37632, 1, 672, 12))
    assert_size_stride(convolution_9, (8, 12, 56, 56), (37632, 1, 672, 12))
    assert_size_stride(squeeze_28, (12, ), (1, ))
    assert_size_stride(convolution_10, (8, 16, 56, 56), (50176, 1, 896, 16))
    assert_size_stride(squeeze_31, (16, ), (1, ))
    assert_size_stride(add_55, (8, 16, 56, 56), (50176, 1, 896, 16))
    assert_size_stride(convolution_11, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(squeeze_34, (24, ), (1, ))
    assert_size_stride(slice_22, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_12, (8, 36, 56, 56), (112896, 1, 2016, 36))
    assert_size_stride(squeeze_37, (36, ), (1, ))
    assert_size_stride(relu_5, (8, 36, 56, 56), (112896, 1, 2016, 36))
    assert_size_stride(convolution_13, (8, 36, 56, 56), (112896, 1, 2016, 36))
    assert_size_stride(squeeze_40, (36, ), (1, ))
    assert_size_stride(slice_25, (8, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_14, (8, 12, 56, 56), (37632, 1, 672, 12))
    assert_size_stride(squeeze_43, (12, ), (1, ))
    assert_size_stride(add_76, (8, 12, 56, 56), (37632, 1, 672, 12))
    assert_size_stride(convolution_15, (8, 12, 56, 56), (37632, 1, 672, 12))
    assert_size_stride(squeeze_46, (12, ), (1, ))
    assert_size_stride(slice_33, (8, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_16, (8, 36, 56, 56), (112896, 1, 2016, 36))
    assert_size_stride(squeeze_49, (36, ), (1, ))
    assert_size_stride(relu_7, (8, 36, 56, 56), (112896, 1, 2016, 36))
    assert_size_stride(convolution_17, (8, 36, 56, 56), (112896, 1, 2016, 36))
    assert_size_stride(squeeze_52, (36, ), (1, ))
    assert_size_stride(slice_36, (8, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_18, (8, 72, 28, 28), (56448, 1, 2016, 72))
    assert_size_stride(squeeze_55, (72, ), (1, ))
    assert_size_stride(add_97, (8, 72, 28, 28), (56448, 1, 2016, 72))
    assert_size_stride(mean, (8, 72, 1, 1), (72, 1, 72, 72))
    assert_size_stride(relu_9, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(div, (8, 72, 1, 1), (72, 1, 72, 72))
    assert_size_stride(mul_133, (8, 72, 28, 28), (56448, 1, 2016, 72))
    assert_size_stride(convolution_21, (8, 20, 28, 28), (15680, 1, 560, 20))
    assert_size_stride(squeeze_58, (20, ), (1, ))
    assert_size_stride(add_103, (8, 20, 28, 28), (15680, 1, 560, 20))
    assert_size_stride(convolution_22, (8, 20, 28, 28), (15680, 1, 560, 20))
    assert_size_stride(squeeze_61, (20, ), (1, ))
    assert_size_stride(convolution_23, (8, 24, 28, 28), (18816, 1, 672, 24))
    assert_size_stride(squeeze_64, (24, ), (1, ))
    assert_size_stride(add_113, (8, 24, 28, 28), (18816, 1, 672, 24))
    assert_size_stride(convolution_24, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(squeeze_67, (40, ), (1, ))
    assert_size_stride(slice_44, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_25, (8, 60, 28, 28), (47040, 1, 1680, 60))
    assert_size_stride(squeeze_70, (60, ), (1, ))
    assert_size_stride(relu_10, (8, 60, 28, 28), (47040, 1, 1680, 60))
    assert_size_stride(convolution_26, (8, 60, 28, 28), (47040, 1, 1680, 60))
    assert_size_stride(squeeze_73, (60, ), (1, ))
    assert_size_stride(cat_8, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(mean_1, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(relu_12, (8, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_1, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(mul_176, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_29, (8, 20, 28, 28), (15680, 1, 560, 20))
    assert_size_stride(squeeze_76, (20, ), (1, ))
    assert_size_stride(add_135, (8, 20, 28, 28), (15680, 1, 560, 20))
    assert_size_stride(convolution_30, (8, 20, 28, 28), (15680, 1, 560, 20))
    assert_size_stride(squeeze_79, (20, ), (1, ))
    assert_size_stride(slice_55, (8, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_31, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_82, (120, ), (1, ))
    assert_size_stride(relu_13, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_32, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(squeeze_85, (120, ), (1, ))
    assert_size_stride(slice_58, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(convolution_33, (8, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(squeeze_88, (240, ), (1, ))
    assert_size_stride(add_156, (8, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(convolution_34, (8, 40, 14, 14), (7840, 1, 560, 40))
    assert_size_stride(squeeze_91, (40, ), (1, ))
    assert_size_stride(add_161, (8, 40, 14, 14), (7840, 1, 560, 40))
    assert_size_stride(convolution_35, (8, 40, 14, 14), (7840, 1, 560, 40))
    assert_size_stride(squeeze_94, (40, ), (1, ))
    assert_size_stride(convolution_36, (8, 40, 14, 14), (7840, 1, 560, 40))
    assert_size_stride(squeeze_97, (40, ), (1, ))
    assert_size_stride(add_171, (8, 40, 14, 14), (7840, 1, 560, 40))
    assert_size_stride(convolution_37, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(squeeze_100, (80, ), (1, ))
    assert_size_stride(slice_66, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_38, (8, 100, 14, 14), (19600, 1, 1400, 100))
    assert_size_stride(squeeze_103, (100, ), (1, ))
    assert_size_stride(relu_15, (8, 100, 14, 14), (19600, 1, 1400, 100))
    assert_size_stride(convolution_39, (8, 100, 14, 14), (19600, 1, 1400, 100))
    assert_size_stride(squeeze_106, (100, ), (1, ))
    assert_size_stride(slice_69, (8, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(convolution_40, (8, 40, 14, 14), (7840, 1, 560, 40))
    assert_size_stride(squeeze_109, (40, ), (1, ))
    assert_size_stride(add_192, (8, 40, 14, 14), (7840, 1, 560, 40))
    assert_size_stride(convolution_41, (8, 40, 14, 14), (7840, 1, 560, 40))
    assert_size_stride(squeeze_112, (40, ), (1, ))
    assert_size_stride(slice_77, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_42, (8, 92, 14, 14), (18032, 1, 1288, 92))
    assert_size_stride(squeeze_115, (92, ), (1, ))
    assert_size_stride(relu_17, (8, 92, 14, 14), (18032, 1, 1288, 92))
    assert_size_stride(convolution_43, (8, 92, 14, 14), (18032, 1, 1288, 92))
    assert_size_stride(squeeze_118, (92, ), (1, ))
    assert_size_stride(slice_80, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(convolution_44, (8, 40, 14, 14), (7840, 1, 560, 40))
    assert_size_stride(squeeze_121, (40, ), (1, ))
    assert_size_stride(add_213, (8, 40, 14, 14), (7840, 1, 560, 40))
    assert_size_stride(convolution_45, (8, 40, 14, 14), (7840, 1, 560, 40))
    assert_size_stride(squeeze_124, (40, ), (1, ))
    assert_size_stride(slice_88, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_46, (8, 92, 14, 14), (18032, 1, 1288, 92))
    assert_size_stride(squeeze_127, (92, ), (1, ))
    assert_size_stride(relu_19, (8, 92, 14, 14), (18032, 1, 1288, 92))
    assert_size_stride(convolution_47, (8, 92, 14, 14), (18032, 1, 1288, 92))
    assert_size_stride(squeeze_130, (92, ), (1, ))
    assert_size_stride(slice_91, (8, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(convolution_48, (8, 40, 14, 14), (7840, 1, 560, 40))
    assert_size_stride(squeeze_133, (40, ), (1, ))
    assert_size_stride(add_234, (8, 40, 14, 14), (7840, 1, 560, 40))
    assert_size_stride(convolution_49, (8, 40, 14, 14), (7840, 1, 560, 40))
    assert_size_stride(squeeze_136, (40, ), (1, ))
    assert_size_stride(slice_99, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_50, (8, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(squeeze_139, (240, ), (1, ))
    assert_size_stride(relu_21, (8, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(convolution_51, (8, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(squeeze_142, (240, ), (1, ))
    assert_size_stride(cat_18, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_2, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(relu_23, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(div_2, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_338, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_54, (8, 56, 14, 14), (10976, 1, 784, 56))
    assert_size_stride(squeeze_145, (56, ), (1, ))
    assert_size_stride(add_256, (8, 56, 14, 14), (10976, 1, 784, 56))
    assert_size_stride(convolution_55, (8, 56, 14, 14), (10976, 1, 784, 56))
    assert_size_stride(squeeze_148, (56, ), (1, ))
    assert_size_stride(convolution_56, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(squeeze_151, (80, ), (1, ))
    assert_size_stride(add_266, (8, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_57, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(squeeze_154, (112, ), (1, ))
    assert_size_stride(slice_110, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(convolution_58, (8, 336, 14, 14), (65856, 1, 4704, 336))
    assert_size_stride(squeeze_157, (336, ), (1, ))
    assert_size_stride(relu_24, (8, 336, 14, 14), (65856, 1, 4704, 336))
    assert_size_stride(convolution_59, (8, 336, 14, 14), (65856, 1, 4704, 336))
    assert_size_stride(squeeze_160, (336, ), (1, ))
    assert_size_stride(cat_20, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(mean_3, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(relu_26, (8, 168, 1, 1), (168, 1, 168, 168))
    assert_size_stride(div_3, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_381, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_62, (8, 56, 14, 14), (10976, 1, 784, 56))
    assert_size_stride(squeeze_163, (56, ), (1, ))
    assert_size_stride(add_288, (8, 56, 14, 14), (10976, 1, 784, 56))
    assert_size_stride(convolution_63, (8, 56, 14, 14), (10976, 1, 784, 56))
    assert_size_stride(squeeze_166, (56, ), (1, ))
    assert_size_stride(slice_121, (8, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(convolution_64, (8, 336, 14, 14), (65856, 1, 4704, 336))
    assert_size_stride(squeeze_169, (336, ), (1, ))
    assert_size_stride(relu_27, (8, 336, 14, 14), (65856, 1, 4704, 336))
    assert_size_stride(convolution_65, (8, 336, 14, 14), (65856, 1, 4704, 336))
    assert_size_stride(squeeze_172, (336, ), (1, ))
    assert_size_stride(slice_124, (8, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_66, (8, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(squeeze_175, (672, ), (1, ))
    assert_size_stride(add_309, (8, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(mean_4, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(relu_29, (8, 168, 1, 1), (168, 1, 168, 168))
    assert_size_stride(div_4, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_417, (8, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(convolution_69, (8, 80, 7, 7), (3920, 1, 560, 80))
    assert_size_stride(squeeze_178, (80, ), (1, ))
    assert_size_stride(add_315, (8, 80, 7, 7), (3920, 1, 560, 80))
    assert_size_stride(convolution_70, (8, 80, 7, 7), (3920, 1, 560, 80))
    assert_size_stride(squeeze_181, (80, ), (1, ))
    assert_size_stride(convolution_71, (8, 112, 7, 7), (5488, 1, 784, 112))
    assert_size_stride(squeeze_184, (112, ), (1, ))
    assert_size_stride(add_325, (8, 112, 7, 7), (5488, 1, 784, 112))
    assert_size_stride(convolution_72, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(squeeze_187, (160, ), (1, ))
    assert_size_stride(slice_132, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_73, (8, 480, 7, 7), (23520, 1, 3360, 480))
    assert_size_stride(squeeze_190, (480, ), (1, ))
    assert_size_stride(relu_30, (8, 480, 7, 7), (23520, 1, 3360, 480))
    assert_size_stride(convolution_74, (8, 480, 7, 7), (23520, 1, 3360, 480))
    assert_size_stride(squeeze_193, (480, ), (1, ))
    assert_size_stride(slice_135, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_75, (8, 80, 7, 7), (3920, 1, 560, 80))
    assert_size_stride(squeeze_196, (80, ), (1, ))
    assert_size_stride(add_346, (8, 80, 7, 7), (3920, 1, 560, 80))
    assert_size_stride(convolution_76, (8, 80, 7, 7), (3920, 1, 560, 80))
    assert_size_stride(squeeze_199, (80, ), (1, ))
    assert_size_stride(slice_143, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_77, (8, 480, 7, 7), (23520, 1, 3360, 480))
    assert_size_stride(squeeze_202, (480, ), (1, ))
    assert_size_stride(relu_32, (8, 480, 7, 7), (23520, 1, 3360, 480))
    assert_size_stride(convolution_78, (8, 480, 7, 7), (23520, 1, 3360, 480))
    assert_size_stride(squeeze_205, (480, ), (1, ))
    assert_size_stride(cat_26, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(mean_5, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(relu_34, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(div_5, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(mul_488, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_81, (8, 80, 7, 7), (3920, 1, 560, 80))
    assert_size_stride(squeeze_208, (80, ), (1, ))
    assert_size_stride(add_368, (8, 80, 7, 7), (3920, 1, 560, 80))
    assert_size_stride(convolution_82, (8, 80, 7, 7), (3920, 1, 560, 80))
    assert_size_stride(squeeze_211, (80, ), (1, ))
    assert_size_stride(slice_154, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_83, (8, 480, 7, 7), (23520, 1, 3360, 480))
    assert_size_stride(squeeze_214, (480, ), (1, ))
    assert_size_stride(relu_35, (8, 480, 7, 7), (23520, 1, 3360, 480))
    assert_size_stride(convolution_84, (8, 480, 7, 7), (23520, 1, 3360, 480))
    assert_size_stride(squeeze_217, (480, ), (1, ))
    assert_size_stride(slice_157, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_85, (8, 80, 7, 7), (3920, 1, 560, 80))
    assert_size_stride(squeeze_220, (80, ), (1, ))
    assert_size_stride(add_389, (8, 80, 7, 7), (3920, 1, 560, 80))
    assert_size_stride(convolution_86, (8, 80, 7, 7), (3920, 1, 560, 80))
    assert_size_stride(squeeze_223, (80, ), (1, ))
    assert_size_stride(slice_165, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_87, (8, 480, 7, 7), (23520, 1, 3360, 480))
    assert_size_stride(squeeze_226, (480, ), (1, ))
    assert_size_stride(relu_37, (8, 480, 7, 7), (23520, 1, 3360, 480))
    assert_size_stride(convolution_88, (8, 480, 7, 7), (23520, 1, 3360, 480))
    assert_size_stride(squeeze_229, (480, ), (1, ))
    assert_size_stride(cat_30, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(mean_6, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(relu_39, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(div_6, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(mul_545, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_91, (8, 80, 7, 7), (3920, 1, 560, 80))
    assert_size_stride(squeeze_232, (80, ), (1, ))
    assert_size_stride(add_411, (8, 80, 7, 7), (3920, 1, 560, 80))
    assert_size_stride(convolution_92, (8, 80, 7, 7), (3920, 1, 560, 80))
    assert_size_stride(squeeze_235, (80, ), (1, ))
    assert_size_stride(slice_176, (8, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_93, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_238, (960, ), (1, ))
    assert_size_stride(mean_7, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(view_1, (8, 1280), (1280, 1))
    assert_size_stride(permute_1, (1000, 1280), (1280, 1))
    assert_size_stride(le, (8, 1280, 1, 1), (1280, 1, 1280, 1280))
    assert_size_stride(le_1, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(unsqueeze_322, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_334, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_346, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(bitwise_and, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(le_3, (8, 480, 7, 7), (23520, 1, 3360, 480))
    assert_size_stride(unsqueeze_358, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_370, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_382, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_394, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(le_5, (8, 480, 7, 7), (23520, 1, 3360, 480))
    assert_size_stride(unsqueeze_406, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_418, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_430, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_442, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(bitwise_and_1, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(le_8, (8, 480, 7, 7), (23520, 1, 3360, 480))
    assert_size_stride(unsqueeze_454, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_466, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_478, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_490, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(le_10, (8, 480, 7, 7), (23520, 1, 3360, 480))
    assert_size_stride(unsqueeze_502, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_514, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_526, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_538, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(unsqueeze_550, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_562, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(bitwise_and_2, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(unsqueeze_574, (1, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(le_13, (8, 336, 14, 14), (65856, 1, 4704, 336))
    assert_size_stride(unsqueeze_586, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(unsqueeze_598, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(unsqueeze_610, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(unsqueeze_622, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(bitwise_and_3, (8, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(le_16, (8, 336, 14, 14), (65856, 1, 4704, 336))
    assert_size_stride(unsqueeze_634, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(unsqueeze_646, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(unsqueeze_658, (1, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(unsqueeze_670, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_682, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(unsqueeze_694, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(bitwise_and_4, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(le_19, (8, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(unsqueeze_706, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(unsqueeze_718, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(unsqueeze_730, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(unsqueeze_742, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(le_21, (8, 92, 14, 14), (18032, 1, 1288, 92))
    assert_size_stride(unsqueeze_754, (1, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(unsqueeze_766, (1, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(unsqueeze_778, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(unsqueeze_790, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(le_23, (8, 92, 14, 14), (18032, 1, 1288, 92))
    assert_size_stride(unsqueeze_802, (1, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(unsqueeze_814, (1, 92, 1, 1), (92, 1, 1, 1))
    assert_size_stride(unsqueeze_826, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(unsqueeze_838, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(le_25, (8, 100, 14, 14), (19600, 1, 1400, 100))
    assert_size_stride(unsqueeze_850, (1, 100, 1, 1), (100, 1, 1, 1))
    assert_size_stride(unsqueeze_862, (1, 100, 1, 1), (100, 1, 1, 1))
    assert_size_stride(unsqueeze_874, (1, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(unsqueeze_886, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(unsqueeze_898, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(unsqueeze_910, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(unsqueeze_922, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(le_27, (8, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(unsqueeze_934, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_946, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_958, (1, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(unsqueeze_970, (1, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(bitwise_and_5, (8, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(le_30, (8, 60, 28, 28), (47040, 1, 1680, 60))
    assert_size_stride(unsqueeze_982, (1, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(unsqueeze_994, (1, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(unsqueeze_1006, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(unsqueeze_1018, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_1030, (1, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(unsqueeze_1042, (1, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(bitwise_and_6, (8, 72, 1, 1), (72, 1, 72, 72))
    assert_size_stride(unsqueeze_1054, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(le_33, (8, 36, 56, 56), (112896, 1, 2016, 36))
    assert_size_stride(unsqueeze_1066, (1, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(unsqueeze_1078, (1, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(unsqueeze_1090, (1, 12, 1, 1), (12, 1, 1, 1))
    assert_size_stride(unsqueeze_1102, (1, 12, 1, 1), (12, 1, 1, 1))
    assert_size_stride(le_35, (8, 36, 56, 56), (112896, 1, 2016, 36))
    assert_size_stride(unsqueeze_1114, (1, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(unsqueeze_1126, (1, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(unsqueeze_1138, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_1150, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(unsqueeze_1162, (1, 12, 1, 1), (12, 1, 1, 1))
    assert_size_stride(unsqueeze_1174, (1, 12, 1, 1), (12, 1, 1, 1))
    assert_size_stride(unsqueeze_1186, (1, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(le_37, (8, 24, 112, 112), (301056, 1, 2688, 24))
    assert_size_stride(unsqueeze_1198, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_1210, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(unsqueeze_1222, (1, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(unsqueeze_1234, (1, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(le_39, (8, 8, 112, 112), (100352, 1, 896, 8))
    assert_size_stride(unsqueeze_1246, (1, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(unsqueeze_1258, (1, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(unsqueeze_1270, (1, 16, 1, 1), (16, 1, 1, 1))
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
    buf3 = reinterpret_tensor(buf0, (8, 1280, 1, 1), (1280, 1, 1280, 1280), 0); del buf0  # reuse
    cpp_fused_convolution_backward_sum_threshold_backward_0(c_void_p(buf3.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf2.data_ptr()))
    del le
    del tangents_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf4 = aten.convolution_backward(buf3, mean_7, primals_271, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf3
    del mean_7
    del primals_271
    buf5 = buf4[0]
    buf6 = buf4[1]
    buf7 = buf4[2]
    del buf4
    buf8 = empty((960, ), device='cpu', dtype=torch.float32)
    buf9 = empty((960, ), device='cpu', dtype=torch.float32)
    buf10 = empty((960, ), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_1(c_void_p(le_1.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(convolution_93.data_ptr()), c_void_p(unsqueeze_322.data_ptr()), c_void_p(squeeze_238.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()))
    del buf9
    del convolution_93
    del le_1
    del primals_1
    del squeeze_238
    del unsqueeze_322
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
    buf12 = aten.convolution_backward(buf11, slice_176, primals_270, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf11
    del primals_270
    del slice_176
    buf13 = buf12[0]
    buf14 = buf12[1]
    del buf12
    buf16 = empty((8, 160, 7, 7), device='cpu', dtype=torch.float32)
    buf17 = empty((62720, ), device='cpu', dtype=torch.float32)
    buf20 = empty((8, 160, 7, 7), device='cpu', dtype=torch.float32)
    buf22 = empty((80, ), device='cpu', dtype=torch.float32)
    buf23 = empty((80, ), device='cpu', dtype=torch.float32)
    buf24 = empty((80, ), device='cpu', dtype=torch.float32)
    buf25 = empty((8, 80, 7, 7), device='cpu', dtype=torch.float32)
    cpp_fused_as_strided_scatter_convolution_backward_copy_native_batch_norm_backward_2(c_void_p(buf13.data_ptr()), c_void_p(convolution_92.data_ptr()), c_void_p(unsqueeze_334.data_ptr()), c_void_p(squeeze_235.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()))
    del convolution_92
    del primals_268
    del squeeze_235
    del unsqueeze_334
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf26 = aten.convolution_backward(buf25, add_411, primals_267, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False])
    del add_411
    del primals_267
    buf27 = buf26[0]
    buf28 = buf26[1]
    del buf26
    buf29 = buf23; del buf23  # reuse
    buf30 = empty((80, ), device='cpu', dtype=torch.float32)
    buf31 = empty((80, ), device='cpu', dtype=torch.float32)
    buf32 = buf25; del buf25  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_3(c_void_p(buf20.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(convolution_91.data_ptr()), c_void_p(unsqueeze_346.data_ptr()), c_void_p(squeeze_232.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()))
    del buf27
    del convolution_91
    del primals_265
    del squeeze_232
    del unsqueeze_346
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf33 = aten.convolution_backward(buf32, mul_545, primals_264, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_545
    del primals_264
    buf34 = buf33[0]
    buf35 = buf33[1]
    del buf33
    buf36 = reinterpret_tensor(buf5, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf5  # reuse
    buf37 = reinterpret_tensor(buf36, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf36  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_threshold_backward_4(c_void_p(buf37.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(cat_30.data_ptr()), c_void_p(bitwise_and.data_ptr()))
    del bitwise_and
    del cat_30
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.threshold_backward]
    buf38 = aten.convolution_backward(buf37, relu_39, primals_262, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf37
    del primals_262
    buf39 = buf38[0]
    buf40 = buf38[1]
    buf41 = buf38[2]
    del buf38
    buf42 = buf39; del buf39  # reuse
    cpp_fused_convolution_backward_threshold_backward_5(c_void_p(buf42.data_ptr()), c_void_p(relu_39.data_ptr()))
    del relu_39
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf43 = aten.convolution_backward(buf42, mean_6, primals_260, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf42
    del mean_6
    del primals_260
    buf44 = buf43[0]
    buf45 = buf43[1]
    buf46 = buf43[2]
    del buf43
    buf47 = empty((480, ), device='cpu', dtype=torch.float32)
    buf48 = empty((480, ), device='cpu', dtype=torch.float32)
    buf49 = empty_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.float32)
    buf50 = buf48; del buf48  # reuse
    buf51 = buf49; del buf49  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6(c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(le_3.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(div_6.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(convolution_88.data_ptr()), c_void_p(unsqueeze_358.data_ptr()), c_void_p(squeeze_229.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(buf47.data_ptr()))
    del convolution_88
    del le_3
    del primals_258
    del squeeze_229
    del unsqueeze_358
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf52 = aten.convolution_backward(buf51, relu_37, primals_257, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
    del buf51
    del primals_257
    buf53 = buf52[0]
    buf54 = buf52[1]
    del buf52
    buf55 = buf53; del buf53  # reuse
    buf56 = empty((480, ), device='cpu', dtype=torch.float32)
    buf57 = empty((480, ), device='cpu', dtype=torch.float32)
    buf58 = empty((480, ), device='cpu', dtype=torch.float32)
    buf59 = buf55; del buf55  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_7(c_void_p(buf59.data_ptr()), c_void_p(relu_37.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(div_6.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(convolution_87.data_ptr()), c_void_p(unsqueeze_370.data_ptr()), c_void_p(squeeze_226.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()))
    del buf34
    del convolution_87
    del div_6
    del primals_255
    del relu_37
    del squeeze_226
    del unsqueeze_370
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf60 = aten.convolution_backward(buf59, slice_165, primals_254, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_254
    del slice_165
    buf61 = buf60[0]
    buf62 = buf60[1]
    del buf60
    buf63 = buf20; del buf20  # reuse
    buf64 = reinterpret_tensor(buf16, (62720, ), (1, ), 0); del buf16  # reuse
    buf67 = reinterpret_tensor(buf13, (8, 160, 7, 7), (7840, 49, 7, 1), 0); del buf13  # reuse
    buf69 = buf30; del buf30  # reuse
    buf70 = empty((80, ), device='cpu', dtype=torch.float32)
    buf71 = empty((80, ), device='cpu', dtype=torch.float32)
    buf72 = buf32; del buf32  # reuse
    cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_8(c_void_p(buf17.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(convolution_86.data_ptr()), c_void_p(unsqueeze_382.data_ptr()), c_void_p(squeeze_223.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()))
    del buf17
    del convolution_86
    del primals_252
    del squeeze_223
    del unsqueeze_382
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf73 = aten.convolution_backward(buf72, add_389, primals_251, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False])
    del add_389
    del primals_251
    buf74 = buf73[0]
    buf75 = buf73[1]
    del buf73
    buf76 = buf70; del buf70  # reuse
    buf77 = empty((80, ), device='cpu', dtype=torch.float32)
    buf78 = empty((80, ), device='cpu', dtype=torch.float32)
    buf79 = buf72; del buf72  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_9(c_void_p(buf67.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(convolution_85.data_ptr()), c_void_p(unsqueeze_394.data_ptr()), c_void_p(squeeze_220.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    del buf74
    del convolution_85
    del primals_249
    del squeeze_220
    del unsqueeze_394
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf80 = aten.convolution_backward(buf79, slice_157, primals_248, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_248
    del slice_157
    buf81 = buf80[0]
    buf82 = buf80[1]
    del buf80
    buf83 = buf57; del buf57  # reuse
    buf84 = empty((480, ), device='cpu', dtype=torch.float32)
    buf85 = empty((480, ), device='cpu', dtype=torch.float32)
    buf86 = buf59; del buf59  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10(c_void_p(le_5.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(convolution_84.data_ptr()), c_void_p(unsqueeze_406.data_ptr()), c_void_p(squeeze_217.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()))
    del convolution_84
    del le_5
    del primals_246
    del squeeze_217
    del unsqueeze_406
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf87 = aten.convolution_backward(buf86, relu_35, primals_245, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
    del buf86
    del primals_245
    buf88 = buf87[0]
    buf89 = buf87[1]
    del buf87
    buf90 = buf84; del buf84  # reuse
    buf91 = empty((480, ), device='cpu', dtype=torch.float32)
    buf92 = buf88; del buf88  # reuse
    buf93 = buf91; del buf91  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_11(c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(relu_35.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(convolution_83.data_ptr()), c_void_p(unsqueeze_418.data_ptr()), c_void_p(squeeze_214.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(buf90.data_ptr()))
    del buf81
    del convolution_83
    del primals_243
    del relu_35
    del squeeze_214
    del unsqueeze_418
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf94 = aten.convolution_backward(buf92, slice_154, primals_242, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_242
    del slice_154
    buf95 = buf94[0]
    buf96 = buf94[1]
    del buf94
    buf97 = buf67; del buf67  # reuse
    buf98 = reinterpret_tensor(buf63, (62720, ), (1, ), 0); del buf63  # reuse
    buf101 = reinterpret_tensor(buf61, (8, 160, 7, 7), (7840, 49, 7, 1), 0); del buf61  # reuse
    buf103 = buf77; del buf77  # reuse
    buf104 = empty((80, ), device='cpu', dtype=torch.float32)
    buf105 = empty((80, ), device='cpu', dtype=torch.float32)
    buf106 = buf79; del buf79  # reuse
    cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_12(c_void_p(buf64.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(convolution_82.data_ptr()), c_void_p(unsqueeze_430.data_ptr()), c_void_p(squeeze_211.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    del buf64
    del convolution_82
    del primals_240
    del squeeze_211
    del unsqueeze_430
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf107 = aten.convolution_backward(buf106, add_368, primals_239, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False])
    del add_368
    del primals_239
    buf108 = buf107[0]
    buf109 = buf107[1]
    del buf107
    buf110 = buf104; del buf104  # reuse
    buf111 = empty((80, ), device='cpu', dtype=torch.float32)
    buf112 = empty((80, ), device='cpu', dtype=torch.float32)
    buf113 = buf106; del buf106  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_13(c_void_p(buf101.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(convolution_81.data_ptr()), c_void_p(unsqueeze_442.data_ptr()), c_void_p(squeeze_208.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()))
    del buf108
    del convolution_81
    del primals_237
    del squeeze_208
    del unsqueeze_442
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf114 = aten.convolution_backward(buf113, mul_488, primals_236, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_488
    del primals_236
    buf115 = buf114[0]
    buf116 = buf114[1]
    del buf114
    buf117 = reinterpret_tensor(buf44, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf44  # reuse
    buf118 = reinterpret_tensor(buf117, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf117  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_threshold_backward_14(c_void_p(buf118.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(cat_26.data_ptr()), c_void_p(bitwise_and_1.data_ptr()))
    del bitwise_and_1
    del cat_26
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.threshold_backward]
    buf119 = aten.convolution_backward(buf118, relu_34, primals_234, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf118
    del primals_234
    buf120 = buf119[0]
    buf121 = buf119[1]
    buf122 = buf119[2]
    del buf119
    buf123 = buf120; del buf120  # reuse
    cpp_fused_convolution_backward_threshold_backward_15(c_void_p(buf123.data_ptr()), c_void_p(relu_34.data_ptr()))
    del relu_34
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf124 = aten.convolution_backward(buf123, mean_5, primals_232, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf123
    del mean_5
    del primals_232
    buf125 = buf124[0]
    buf126 = buf124[1]
    buf127 = buf124[2]
    del buf124
    buf128 = empty((480, ), device='cpu', dtype=torch.float32)
    buf129 = empty((480, ), device='cpu', dtype=torch.float32)
    buf130 = buf92; del buf92  # reuse
    buf131 = buf129; del buf129  # reuse
    buf132 = buf130; del buf130  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_16(c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(le_8.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(div_5.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(convolution_78.data_ptr()), c_void_p(unsqueeze_454.data_ptr()), c_void_p(squeeze_205.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(buf128.data_ptr()))
    del convolution_78
    del le_8
    del primals_230
    del squeeze_205
    del unsqueeze_454
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf133 = aten.convolution_backward(buf132, relu_32, primals_229, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
    del buf132
    del primals_229
    buf134 = buf133[0]
    buf135 = buf133[1]
    del buf133
    buf136 = buf134; del buf134  # reuse
    buf137 = empty((480, ), device='cpu', dtype=torch.float32)
    buf138 = empty((480, ), device='cpu', dtype=torch.float32)
    buf139 = empty((480, ), device='cpu', dtype=torch.float32)
    buf140 = buf136; del buf136  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_17(c_void_p(buf140.data_ptr()), c_void_p(relu_32.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(div_5.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(convolution_77.data_ptr()), c_void_p(unsqueeze_466.data_ptr()), c_void_p(squeeze_202.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()))
    del buf115
    del buf125
    del convolution_77
    del div_5
    del primals_227
    del relu_32
    del squeeze_202
    del unsqueeze_466
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf141 = aten.convolution_backward(buf140, slice_143, primals_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_226
    del slice_143
    buf142 = buf141[0]
    buf143 = buf141[1]
    del buf141
    buf144 = buf101; del buf101  # reuse
    buf145 = reinterpret_tensor(buf97, (62720, ), (1, ), 0); del buf97  # reuse
    buf148 = reinterpret_tensor(buf95, (8, 160, 7, 7), (7840, 49, 7, 1), 0); del buf95  # reuse
    buf150 = buf111; del buf111  # reuse
    buf151 = empty((80, ), device='cpu', dtype=torch.float32)
    buf152 = empty((80, ), device='cpu', dtype=torch.float32)
    buf153 = buf113; del buf113  # reuse
    cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_18(c_void_p(buf98.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(convolution_76.data_ptr()), c_void_p(unsqueeze_478.data_ptr()), c_void_p(squeeze_199.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()))
    del convolution_76
    del primals_224
    del squeeze_199
    del unsqueeze_478
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf154 = aten.convolution_backward(buf153, add_346, primals_223, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False])
    del add_346
    del primals_223
    buf155 = buf154[0]
    buf156 = buf154[1]
    del buf154
    buf157 = buf151; del buf151  # reuse
    buf158 = empty((80, ), device='cpu', dtype=torch.float32)
    buf159 = empty((80, ), device='cpu', dtype=torch.float32)
    buf160 = buf153; del buf153  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_19(c_void_p(buf148.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(convolution_75.data_ptr()), c_void_p(unsqueeze_490.data_ptr()), c_void_p(squeeze_196.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()))
    del buf155
    del convolution_75
    del primals_221
    del squeeze_196
    del unsqueeze_490
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf161 = aten.convolution_backward(buf160, slice_135, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_220
    del slice_135
    buf162 = buf161[0]
    buf163 = buf161[1]
    del buf161
    buf164 = buf138; del buf138  # reuse
    buf165 = empty((480, ), device='cpu', dtype=torch.float32)
    buf166 = empty((480, ), device='cpu', dtype=torch.float32)
    buf167 = buf140; del buf140  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_20(c_void_p(le_10.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(convolution_74.data_ptr()), c_void_p(unsqueeze_502.data_ptr()), c_void_p(squeeze_193.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()))
    del convolution_74
    del le_10
    del primals_218
    del squeeze_193
    del unsqueeze_502
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf168 = aten.convolution_backward(buf167, relu_30, primals_217, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
    del buf167
    del primals_217
    buf169 = buf168[0]
    buf170 = buf168[1]
    del buf168
    buf171 = buf165; del buf165  # reuse
    buf172 = empty((480, ), device='cpu', dtype=torch.float32)
    buf173 = buf169; del buf169  # reuse
    buf174 = buf172; del buf172  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_21(c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(relu_30.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(convolution_73.data_ptr()), c_void_p(unsqueeze_514.data_ptr()), c_void_p(squeeze_190.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(buf171.data_ptr()))
    del convolution_73
    del primals_215
    del relu_30
    del squeeze_190
    del unsqueeze_514
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf175 = aten.convolution_backward(buf173, slice_132, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf173
    del primals_214
    del slice_132
    buf176 = buf175[0]
    buf177 = buf175[1]
    del buf175
    buf178 = buf148; del buf148  # reuse
    buf179 = buf98; del buf98  # reuse
    buf182 = buf144; del buf144  # reuse
    buf184 = empty((160, ), device='cpu', dtype=torch.float32)
    buf185 = empty((160, ), device='cpu', dtype=torch.float32)
    buf186 = empty((160, ), device='cpu', dtype=torch.float32)
    buf187 = reinterpret_tensor(buf142, (8, 160, 7, 7), (7840, 49, 7, 1), 0); del buf142  # reuse
    cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_22(c_void_p(buf145.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(convolution_72.data_ptr()), c_void_p(unsqueeze_526.data_ptr()), c_void_p(squeeze_187.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()))
    del buf145
    del buf176
    del buf178
    del buf179
    del buf185
    del convolution_72
    del primals_212
    del squeeze_187
    del unsqueeze_526
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf188 = aten.convolution_backward(buf187, add_325, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_325
    del buf187
    del primals_211
    buf189 = buf188[0]
    buf190 = buf188[1]
    del buf188
    buf191 = empty((112, ), device='cpu', dtype=torch.float32)
    buf192 = empty((112, ), device='cpu', dtype=torch.float32)
    buf193 = empty((112, ), device='cpu', dtype=torch.float32)
    buf194 = buf189; del buf189  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_23(c_void_p(buf194.data_ptr()), c_void_p(convolution_71.data_ptr()), c_void_p(unsqueeze_538.data_ptr()), c_void_p(squeeze_184.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()))
    del convolution_71
    del primals_209
    del squeeze_184
    del unsqueeze_538
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf195 = aten.convolution_backward(buf194, slice_121, primals_208, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 112, [True, True, False])
    del buf194
    del primals_208
    buf196 = buf195[0]
    buf197 = buf195[1]
    del buf195
    buf198 = buf158; del buf158  # reuse
    buf199 = empty((80, ), device='cpu', dtype=torch.float32)
    buf200 = empty((80, ), device='cpu', dtype=torch.float32)
    buf201 = buf160; del buf160  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_24(c_void_p(buf182.data_ptr()), c_void_p(convolution_70.data_ptr()), c_void_p(unsqueeze_550.data_ptr()), c_void_p(squeeze_181.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()))
    del convolution_70
    del primals_206
    del squeeze_181
    del unsqueeze_550
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf202 = aten.convolution_backward(buf201, add_315, primals_205, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False])
    del add_315
    del primals_205
    buf203 = buf202[0]
    buf204 = buf202[1]
    del buf202
    buf205 = buf199; del buf199  # reuse
    buf206 = empty((80, ), device='cpu', dtype=torch.float32)
    buf207 = empty((80, ), device='cpu', dtype=torch.float32)
    buf208 = buf201; del buf201  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_25(c_void_p(buf182.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(convolution_69.data_ptr()), c_void_p(unsqueeze_562.data_ptr()), c_void_p(squeeze_178.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()))
    del buf203
    del convolution_69
    del primals_203
    del squeeze_178
    del unsqueeze_562
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf209 = aten.convolution_backward(buf208, mul_417, primals_202, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf208
    del mul_417
    del primals_202
    buf210 = buf209[0]
    buf211 = buf209[1]
    del buf209
    buf212 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cpu', dtype=torch.float32)
    buf213 = reinterpret_tensor(buf212, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf212  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_threshold_backward_26(c_void_p(buf213.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(add_309.data_ptr()), c_void_p(bitwise_and_2.data_ptr()))
    del add_309
    del bitwise_and_2
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.threshold_backward]
    buf214 = aten.convolution_backward(buf213, relu_29, primals_200, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf213
    del primals_200
    buf215 = buf214[0]
    buf216 = buf214[1]
    buf217 = buf214[2]
    del buf214
    buf218 = buf215; del buf215  # reuse
    cpp_fused_convolution_backward_threshold_backward_27(c_void_p(buf218.data_ptr()), c_void_p(relu_29.data_ptr()))
    del relu_29
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf219 = aten.convolution_backward(buf218, mean_4, primals_198, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf218
    del mean_4
    del primals_198
    buf220 = buf219[0]
    buf221 = buf219[1]
    buf222 = buf219[2]
    del buf219
    buf223 = empty((672, ), device='cpu', dtype=torch.float32)
    buf224 = empty((672, ), device='cpu', dtype=torch.float32)
    buf225 = buf210; del buf210  # reuse
    buf226 = buf224; del buf224  # reuse
    cpp_fused_add_div_mul_native_batch_norm_backward_28(c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(div_4.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(convolution_66.data_ptr()), c_void_p(unsqueeze_574.data_ptr()), c_void_p(squeeze_175.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(buf223.data_ptr()))
    del convolution_66
    del div_4
    del primals_196
    del squeeze_175
    del unsqueeze_574
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf227 = aten.convolution_backward(buf225, slice_124, primals_195, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
    del buf225
    del primals_195
    del slice_124
    buf228 = buf227[0]
    buf229 = buf227[1]
    del buf227
    buf230 = empty((336, ), device='cpu', dtype=torch.float32)
    buf231 = empty((336, ), device='cpu', dtype=torch.float32)
    buf232 = empty((336, ), device='cpu', dtype=torch.float32)
    buf233 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_29(c_void_p(le_13.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(convolution_65.data_ptr()), c_void_p(unsqueeze_586.data_ptr()), c_void_p(squeeze_172.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()))
    del convolution_65
    del le_13
    del primals_193
    del squeeze_172
    del unsqueeze_586
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf234 = aten.convolution_backward(buf233, relu_27, primals_192, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False])
    del buf233
    del primals_192
    buf235 = buf234[0]
    buf236 = buf234[1]
    del buf234
    buf237 = buf231; del buf231  # reuse
    buf238 = empty((336, ), device='cpu', dtype=torch.float32)
    buf239 = buf235; del buf235  # reuse
    buf240 = buf238; del buf238  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_30(c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(relu_27.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(convolution_64.data_ptr()), c_void_p(unsqueeze_598.data_ptr()), c_void_p(squeeze_169.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(buf237.data_ptr()))
    del buf228
    del convolution_64
    del primals_190
    del relu_27
    del squeeze_169
    del unsqueeze_598
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf241 = aten.convolution_backward(buf239, slice_121, primals_189, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_189
    del slice_121
    buf242 = buf241[0]
    buf243 = buf241[1]
    del buf241
    buf245 = empty((8, 112, 14, 14), device='cpu', dtype=torch.float32)
    buf246 = empty((175616, ), device='cpu', dtype=torch.float32)
    buf249 = empty((8, 112, 14, 14), device='cpu', dtype=torch.float32)
    buf251 = empty((56, ), device='cpu', dtype=torch.float32)
    buf252 = empty((56, ), device='cpu', dtype=torch.float32)
    buf253 = empty((56, ), device='cpu', dtype=torch.float32)
    buf254 = empty((8, 56, 14, 14), device='cpu', dtype=torch.float32)
    cpp_fused_add_as_strided_scatter_convolution_backward_copy_native_batch_norm_backward_31(c_void_p(buf196.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(convolution_63.data_ptr()), c_void_p(unsqueeze_610.data_ptr()), c_void_p(squeeze_166.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()))
    del convolution_63
    del primals_187
    del squeeze_166
    del unsqueeze_610
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf255 = aten.convolution_backward(buf254, add_288, primals_186, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 56, [True, True, False])
    del add_288
    del primals_186
    buf256 = buf255[0]
    buf257 = buf255[1]
    del buf255
    buf258 = buf252; del buf252  # reuse
    buf259 = empty((56, ), device='cpu', dtype=torch.float32)
    buf260 = empty((56, ), device='cpu', dtype=torch.float32)
    buf261 = buf254; del buf254  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_32(c_void_p(buf249.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(convolution_62.data_ptr()), c_void_p(unsqueeze_622.data_ptr()), c_void_p(squeeze_163.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()))
    del buf256
    del convolution_62
    del primals_184
    del squeeze_163
    del unsqueeze_622
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf262 = aten.convolution_backward(buf261, mul_381, primals_183, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_381
    del primals_183
    buf263 = buf262[0]
    buf264 = buf262[1]
    del buf262
    buf265 = reinterpret_tensor(buf220, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf220  # reuse
    buf266 = reinterpret_tensor(buf265, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf265  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_threshold_backward_33(c_void_p(buf266.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(cat_20.data_ptr()), c_void_p(bitwise_and_3.data_ptr()))
    del bitwise_and_3
    del cat_20
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.threshold_backward]
    buf267 = aten.convolution_backward(buf266, relu_26, primals_181, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf266
    del primals_181
    buf268 = buf267[0]
    buf269 = buf267[1]
    buf270 = buf267[2]
    del buf267
    buf271 = buf268; del buf268  # reuse
    cpp_fused_convolution_backward_threshold_backward_34(c_void_p(buf271.data_ptr()), c_void_p(relu_26.data_ptr()))
    del relu_26
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf272 = aten.convolution_backward(buf271, mean_3, primals_179, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf271
    del mean_3
    del primals_179
    buf273 = buf272[0]
    buf274 = buf272[1]
    buf275 = buf272[2]
    del buf272
    buf276 = empty((336, ), device='cpu', dtype=torch.float32)
    buf277 = empty((336, ), device='cpu', dtype=torch.float32)
    buf278 = buf239; del buf239  # reuse
    buf279 = buf277; del buf277  # reuse
    buf280 = buf278; del buf278  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_35(c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(le_16.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(div_3.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(convolution_59.data_ptr()), c_void_p(unsqueeze_634.data_ptr()), c_void_p(squeeze_160.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(buf276.data_ptr()))
    del convolution_59
    del le_16
    del primals_177
    del squeeze_160
    del unsqueeze_634
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf281 = aten.convolution_backward(buf280, relu_24, primals_176, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False])
    del buf280
    del primals_176
    buf282 = buf281[0]
    buf283 = buf281[1]
    del buf281
    buf284 = buf282; del buf282  # reuse
    buf285 = empty((336, ), device='cpu', dtype=torch.float32)
    buf286 = empty((336, ), device='cpu', dtype=torch.float32)
    buf287 = empty((336, ), device='cpu', dtype=torch.float32)
    buf288 = buf284; del buf284  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_36(c_void_p(buf288.data_ptr()), c_void_p(relu_24.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(div_3.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(convolution_58.data_ptr()), c_void_p(unsqueeze_646.data_ptr()), c_void_p(squeeze_157.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()))
    del buf263
    del buf273
    del buf286
    del convolution_58
    del div_3
    del primals_174
    del relu_24
    del squeeze_157
    del unsqueeze_646
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf289 = aten.convolution_backward(buf288, slice_110, primals_173, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf288
    del primals_173
    del slice_110
    buf290 = buf289[0]
    buf291 = buf289[1]
    del buf289
    buf292 = buf249; del buf249  # reuse
    buf293 = reinterpret_tensor(buf245, (175616, ), (1, ), 0); del buf245  # reuse
    buf296 = reinterpret_tensor(buf242, (8, 112, 14, 14), (21952, 196, 14, 1), 0); del buf242  # reuse
    buf298 = buf192; del buf192  # reuse
    buf299 = empty((112, ), device='cpu', dtype=torch.float32)
    buf300 = empty((112, ), device='cpu', dtype=torch.float32)
    buf301 = reinterpret_tensor(buf196, (8, 112, 14, 14), (21952, 196, 14, 1), 0); del buf196  # reuse
    cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_37(c_void_p(buf246.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(convolution_57.data_ptr()), c_void_p(unsqueeze_658.data_ptr()), c_void_p(squeeze_154.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()))
    del buf246
    del buf290
    del buf292
    del buf293
    del buf299
    del convolution_57
    del primals_171
    del squeeze_154
    del unsqueeze_658
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf302 = aten.convolution_backward(buf301, add_266, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_266
    del buf301
    del primals_170
    buf303 = buf302[0]
    buf304 = buf302[1]
    del buf302
    buf305 = buf206; del buf206  # reuse
    buf306 = empty((80, ), device='cpu', dtype=torch.float32)
    buf307 = empty((80, ), device='cpu', dtype=torch.float32)
    buf308 = buf303; del buf303  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_38(c_void_p(buf308.data_ptr()), c_void_p(convolution_56.data_ptr()), c_void_p(unsqueeze_670.data_ptr()), c_void_p(squeeze_151.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()))
    del convolution_56
    del primals_168
    del squeeze_151
    del unsqueeze_670
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf309 = aten.convolution_backward(buf308, slice_99, primals_167, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False])
    del primals_167
    buf310 = buf309[0]
    buf311 = buf309[1]
    del buf309
    buf312 = buf259; del buf259  # reuse
    buf313 = empty((56, ), device='cpu', dtype=torch.float32)
    buf314 = empty((56, ), device='cpu', dtype=torch.float32)
    buf315 = buf261; del buf261  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_39(c_void_p(buf296.data_ptr()), c_void_p(convolution_55.data_ptr()), c_void_p(unsqueeze_682.data_ptr()), c_void_p(squeeze_148.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()))
    del convolution_55
    del primals_165
    del squeeze_148
    del unsqueeze_682
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf316 = aten.convolution_backward(buf315, add_256, primals_164, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 56, [True, True, False])
    del add_256
    del primals_164
    buf317 = buf316[0]
    buf318 = buf316[1]
    del buf316
    buf319 = buf313; del buf313  # reuse
    buf320 = empty((56, ), device='cpu', dtype=torch.float32)
    buf321 = empty((56, ), device='cpu', dtype=torch.float32)
    buf322 = buf315; del buf315  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_40(c_void_p(buf296.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(unsqueeze_694.data_ptr()), c_void_p(squeeze_145.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()))
    del buf296
    del buf317
    del buf320
    del convolution_54
    del primals_162
    del squeeze_145
    del unsqueeze_694
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf323 = aten.convolution_backward(buf322, mul_338, primals_161, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf322
    del mul_338
    del primals_161
    buf324 = buf323[0]
    buf325 = buf323[1]
    del buf323
    buf326 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cpu', dtype=torch.float32)
    buf327 = reinterpret_tensor(buf326, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf326  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_threshold_backward_41(c_void_p(buf327.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(cat_18.data_ptr()), c_void_p(bitwise_and_4.data_ptr()))
    del bitwise_and_4
    del cat_18
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.threshold_backward]
    buf328 = aten.convolution_backward(buf327, relu_23, primals_159, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf327
    del primals_159
    buf329 = buf328[0]
    buf330 = buf328[1]
    buf331 = buf328[2]
    del buf328
    buf332 = buf329; del buf329  # reuse
    cpp_fused_convolution_backward_threshold_backward_42(c_void_p(buf332.data_ptr()), c_void_p(relu_23.data_ptr()))
    del relu_23
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf333 = aten.convolution_backward(buf332, mean_2, primals_157, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_2
    del primals_157
    buf334 = buf333[0]
    buf335 = buf333[1]
    buf336 = buf333[2]
    del buf333
    buf337 = empty((240, ), device='cpu', dtype=torch.float32)
    buf338 = empty((240, ), device='cpu', dtype=torch.float32)
    buf339 = reinterpret_tensor(buf162, (8, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf162  # reuse
    buf340 = buf338; del buf338  # reuse
    buf341 = buf339; del buf339  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43(c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(le_19.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(div_2.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(unsqueeze_706.data_ptr()), c_void_p(squeeze_142.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(buf337.data_ptr()))
    del convolution_51
    del le_19
    del primals_155
    del squeeze_142
    del unsqueeze_706
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf342 = aten.convolution_backward(buf341, relu_21, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False])
    del buf341
    del primals_154
    buf343 = buf342[0]
    buf344 = buf342[1]
    del buf342
    buf345 = buf343; del buf343  # reuse
    buf346 = empty((240, ), device='cpu', dtype=torch.float32)
    buf347 = empty((240, ), device='cpu', dtype=torch.float32)
    buf348 = empty((240, ), device='cpu', dtype=torch.float32)
    buf349 = buf345; del buf345  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_44(c_void_p(buf349.data_ptr()), c_void_p(relu_21.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(div_2.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(unsqueeze_718.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()))
    del buf334
    del convolution_50
    del div_2
    del primals_152
    del relu_21
    del squeeze_139
    del unsqueeze_718
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf350 = aten.convolution_backward(buf349, slice_99, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf349
    del primals_151
    del slice_99
    buf351 = buf350[0]
    buf352 = buf350[1]
    del buf350
    buf354 = reinterpret_tensor(buf308, (8, 80, 14, 14), (15680, 196, 14, 1), 0); del buf308  # reuse
    buf355 = empty((125440, ), device='cpu', dtype=torch.float32)
    buf358 = empty((8, 80, 14, 14), device='cpu', dtype=torch.float32)
    buf360 = empty((40, ), device='cpu', dtype=torch.float32)
    buf361 = empty((40, ), device='cpu', dtype=torch.float32)
    buf362 = empty((40, ), device='cpu', dtype=torch.float32)
    buf363 = reinterpret_tensor(buf182, (8, 40, 14, 14), (7840, 196, 14, 1), 0); del buf182  # reuse
    cpp_fused_add_as_strided_scatter_convolution_backward_copy_native_batch_norm_backward_45(c_void_p(buf310.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(unsqueeze_730.data_ptr()), c_void_p(squeeze_136.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()))
    del buf310
    del convolution_49
    del primals_149
    del squeeze_136
    del unsqueeze_730
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf364 = aten.convolution_backward(buf363, add_234, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 40, [True, True, False])
    del add_234
    del primals_148
    buf365 = buf364[0]
    buf366 = buf364[1]
    del buf364
    buf367 = buf361; del buf361  # reuse
    buf368 = empty((40, ), device='cpu', dtype=torch.float32)
    buf369 = empty((40, ), device='cpu', dtype=torch.float32)
    buf370 = buf363; del buf363  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_46(c_void_p(buf358.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(unsqueeze_742.data_ptr()), c_void_p(squeeze_133.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()))
    del buf365
    del convolution_48
    del primals_146
    del squeeze_133
    del unsqueeze_742
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf371 = aten.convolution_backward(buf370, slice_91, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_145
    del slice_91
    buf372 = buf371[0]
    buf373 = buf371[1]
    del buf371
    buf374 = empty((92, ), device='cpu', dtype=torch.float32)
    buf375 = empty((92, ), device='cpu', dtype=torch.float32)
    buf376 = empty((92, ), device='cpu', dtype=torch.float32)
    buf377 = empty_strided((8, 92, 14, 14), (18032, 1, 1288, 92), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_47(c_void_p(le_21.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(unsqueeze_754.data_ptr()), c_void_p(squeeze_130.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()))
    del convolution_47
    del le_21
    del primals_143
    del squeeze_130
    del unsqueeze_754
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf378 = aten.convolution_backward(buf377, relu_19, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 92, [True, True, False])
    del buf377
    del primals_142
    buf379 = buf378[0]
    buf380 = buf378[1]
    del buf378
    buf381 = buf375; del buf375  # reuse
    buf382 = empty((92, ), device='cpu', dtype=torch.float32)
    buf383 = buf379; del buf379  # reuse
    buf384 = buf382; del buf382  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_48(c_void_p(buf383.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(relu_19.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(unsqueeze_766.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(buf381.data_ptr()))
    del buf372
    del convolution_46
    del primals_140
    del relu_19
    del squeeze_127
    del unsqueeze_766
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf385 = aten.convolution_backward(buf383, slice_88, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_139
    del slice_88
    buf386 = buf385[0]
    buf387 = buf385[1]
    del buf385
    buf388 = buf358; del buf358  # reuse
    buf389 = reinterpret_tensor(buf354, (125440, ), (1, ), 0); del buf354  # reuse
    buf392 = reinterpret_tensor(buf351, (8, 80, 14, 14), (15680, 196, 14, 1), 0); del buf351  # reuse
    buf394 = buf368; del buf368  # reuse
    buf395 = empty((40, ), device='cpu', dtype=torch.float32)
    buf396 = empty((40, ), device='cpu', dtype=torch.float32)
    buf397 = buf370; del buf370  # reuse
    cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_49(c_void_p(buf355.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(unsqueeze_778.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf397.data_ptr()))
    del buf355
    del convolution_45
    del primals_137
    del squeeze_124
    del unsqueeze_778
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf398 = aten.convolution_backward(buf397, add_213, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 40, [True, True, False])
    del add_213
    del primals_136
    buf399 = buf398[0]
    buf400 = buf398[1]
    del buf398
    buf401 = buf395; del buf395  # reuse
    buf402 = empty((40, ), device='cpu', dtype=torch.float32)
    buf403 = empty((40, ), device='cpu', dtype=torch.float32)
    buf404 = buf397; del buf397  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_50(c_void_p(buf392.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(unsqueeze_790.data_ptr()), c_void_p(squeeze_121.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()))
    del buf399
    del convolution_44
    del primals_134
    del squeeze_121
    del unsqueeze_790
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf405 = aten.convolution_backward(buf404, slice_80, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_133
    del slice_80
    buf406 = buf405[0]
    buf407 = buf405[1]
    del buf405
    buf408 = empty((92, ), device='cpu', dtype=torch.float32)
    buf409 = empty((92, ), device='cpu', dtype=torch.float32)
    buf410 = empty((92, ), device='cpu', dtype=torch.float32)
    buf411 = buf383; del buf383  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_51(c_void_p(le_23.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(unsqueeze_802.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf411.data_ptr()))
    del convolution_43
    del le_23
    del primals_131
    del squeeze_118
    del unsqueeze_802
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf412 = aten.convolution_backward(buf411, relu_17, primals_130, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 92, [True, True, False])
    del buf411
    del primals_130
    buf413 = buf412[0]
    buf414 = buf412[1]
    del buf412
    buf415 = buf409; del buf409  # reuse
    buf416 = empty((92, ), device='cpu', dtype=torch.float32)
    buf417 = buf413; del buf413  # reuse
    buf418 = buf416; del buf416  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_52(c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(unsqueeze_814.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf415.data_ptr()))
    del buf406
    del convolution_42
    del primals_128
    del relu_17
    del squeeze_115
    del unsqueeze_814
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf419 = aten.convolution_backward(buf417, slice_77, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf417
    del primals_127
    del slice_77
    buf420 = buf419[0]
    buf421 = buf419[1]
    del buf419
    buf422 = buf392; del buf392  # reuse
    buf423 = reinterpret_tensor(buf388, (125440, ), (1, ), 0); del buf388  # reuse
    buf426 = reinterpret_tensor(buf386, (8, 80, 14, 14), (15680, 196, 14, 1), 0); del buf386  # reuse
    buf428 = buf402; del buf402  # reuse
    buf429 = empty((40, ), device='cpu', dtype=torch.float32)
    buf430 = empty((40, ), device='cpu', dtype=torch.float32)
    buf431 = buf404; del buf404  # reuse
    cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_53(c_void_p(buf389.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(unsqueeze_826.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()))
    del convolution_41
    del primals_125
    del squeeze_112
    del unsqueeze_826
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf432 = aten.convolution_backward(buf431, add_192, primals_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 40, [True, True, False])
    del add_192
    del primals_124
    buf433 = buf432[0]
    buf434 = buf432[1]
    del buf432
    buf435 = buf429; del buf429  # reuse
    buf436 = empty((40, ), device='cpu', dtype=torch.float32)
    buf437 = empty((40, ), device='cpu', dtype=torch.float32)
    buf438 = buf431; del buf431  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_54(c_void_p(buf426.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(unsqueeze_838.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf438.data_ptr()))
    del buf433
    del convolution_40
    del primals_122
    del squeeze_109
    del unsqueeze_838
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf439 = aten.convolution_backward(buf438, slice_69, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf438
    del primals_121
    del slice_69
    buf440 = buf439[0]
    buf441 = buf439[1]
    del buf439
    buf442 = empty((100, ), device='cpu', dtype=torch.float32)
    buf443 = empty((100, ), device='cpu', dtype=torch.float32)
    buf444 = empty((100, ), device='cpu', dtype=torch.float32)
    buf445 = empty_strided((8, 100, 14, 14), (19600, 1, 1400, 100), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_55(c_void_p(le_25.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(unsqueeze_850.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()))
    del convolution_39
    del le_25
    del primals_119
    del squeeze_106
    del unsqueeze_850
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf446 = aten.convolution_backward(buf445, relu_15, primals_118, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 100, [True, True, False])
    del buf445
    del primals_118
    buf447 = buf446[0]
    buf448 = buf446[1]
    del buf446
    buf449 = buf443; del buf443  # reuse
    buf450 = empty((100, ), device='cpu', dtype=torch.float32)
    buf451 = buf447; del buf447  # reuse
    buf452 = buf450; del buf450  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_56(c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(unsqueeze_862.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf449.data_ptr()))
    del buf440
    del convolution_38
    del primals_116
    del relu_15
    del squeeze_103
    del unsqueeze_862
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf453 = aten.convolution_backward(buf451, slice_66, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf451
    del primals_115
    del slice_66
    buf454 = buf453[0]
    buf455 = buf453[1]
    del buf453
    buf456 = buf426; del buf426  # reuse
    buf457 = reinterpret_tensor(buf422, (125440, ), (1, ), 0); del buf422  # reuse
    buf460 = reinterpret_tensor(buf420, (8, 80, 14, 14), (15680, 196, 14, 1), 0); del buf420  # reuse
    buf462 = buf306; del buf306  # reuse
    buf463 = empty((80, ), device='cpu', dtype=torch.float32)
    buf464 = empty((80, ), device='cpu', dtype=torch.float32)
    buf465 = reinterpret_tensor(buf389, (8, 80, 14, 14), (15680, 196, 14, 1), 0); del buf389  # reuse
    cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_57(c_void_p(buf423.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(unsqueeze_874.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()))
    del buf423
    del buf454
    del buf456
    del buf457
    del buf463
    del convolution_37
    del primals_113
    del squeeze_100
    del unsqueeze_874
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf466 = aten.convolution_backward(buf465, add_171, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_171
    del buf465
    del primals_112
    buf467 = buf466[0]
    buf468 = buf466[1]
    del buf466
    buf469 = buf436; del buf436  # reuse
    buf470 = empty((40, ), device='cpu', dtype=torch.float32)
    buf471 = empty((40, ), device='cpu', dtype=torch.float32)
    buf472 = buf467; del buf467  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_58(c_void_p(buf472.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(unsqueeze_886.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()))
    del convolution_36
    del primals_110
    del squeeze_97
    del unsqueeze_886
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf473 = aten.convolution_backward(buf472, slice_55, primals_109, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 40, [True, True, False])
    del primals_109
    buf474 = buf473[0]
    buf475 = buf473[1]
    del buf473
    buf476 = buf470; del buf470  # reuse
    buf477 = empty((40, ), device='cpu', dtype=torch.float32)
    buf478 = empty((40, ), device='cpu', dtype=torch.float32)
    buf479 = reinterpret_tensor(buf472, (8, 40, 14, 14), (7840, 196, 14, 1), 0); del buf472  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_59(c_void_p(buf460.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_898.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf479.data_ptr()))
    del convolution_35
    del primals_107
    del squeeze_94
    del unsqueeze_898
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf480 = aten.convolution_backward(buf479, add_161, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 40, [True, True, False])
    del add_161
    del primals_106
    buf481 = buf480[0]
    buf482 = buf480[1]
    del buf480
    buf483 = buf477; del buf477  # reuse
    buf484 = empty((40, ), device='cpu', dtype=torch.float32)
    buf485 = empty((40, ), device='cpu', dtype=torch.float32)
    buf486 = buf479; del buf479  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_60(c_void_p(buf460.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_910.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()))
    del buf481
    del convolution_34
    del primals_104
    del squeeze_91
    del unsqueeze_910
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf487 = aten.convolution_backward(buf486, add_156, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_156
    del buf486
    del primals_103
    buf488 = buf487[0]
    buf489 = buf487[1]
    del buf487
    buf490 = buf347; del buf347  # reuse
    buf491 = empty((240, ), device='cpu', dtype=torch.float32)
    buf492 = empty((240, ), device='cpu', dtype=torch.float32)
    buf493 = buf488; del buf488  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_61(c_void_p(buf493.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_922.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf492.data_ptr()))
    del buf491
    del convolution_33
    del primals_101
    del squeeze_88
    del unsqueeze_922
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf494 = aten.convolution_backward(buf493, slice_58, primals_100, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False])
    del primals_100
    del slice_58
    buf495 = buf494[0]
    buf496 = buf494[1]
    del buf494
    buf497 = empty((120, ), device='cpu', dtype=torch.float32)
    buf498 = empty((120, ), device='cpu', dtype=torch.float32)
    buf499 = empty((120, ), device='cpu', dtype=torch.float32)
    buf500 = reinterpret_tensor(buf324, (8, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf324  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_62(c_void_p(le_27.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(unsqueeze_934.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()))
    del convolution_32
    del le_27
    del primals_98
    del squeeze_85
    del unsqueeze_934
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf501 = aten.convolution_backward(buf500, relu_13, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False])
    del buf500
    del primals_97
    buf502 = buf501[0]
    buf503 = buf501[1]
    del buf501
    buf504 = buf498; del buf498  # reuse
    buf505 = empty((120, ), device='cpu', dtype=torch.float32)
    buf506 = buf502; del buf502  # reuse
    buf507 = buf505; del buf505  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_63(c_void_p(buf506.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(unsqueeze_946.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf504.data_ptr()))
    del buf495
    del convolution_31
    del primals_95
    del relu_13
    del squeeze_82
    del unsqueeze_946
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf508 = aten.convolution_backward(buf506, slice_55, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf506
    del primals_94
    del slice_55
    buf509 = buf508[0]
    buf510 = buf508[1]
    del buf508
    buf512 = empty((8, 40, 28, 28), device='cpu', dtype=torch.float32)
    buf513 = empty((250880, ), device='cpu', dtype=torch.float32)
    buf516 = empty((8, 40, 28, 28), device='cpu', dtype=torch.float32)
    buf518 = empty((20, ), device='cpu', dtype=torch.float32)
    buf519 = empty((20, ), device='cpu', dtype=torch.float32)
    buf520 = empty((20, ), device='cpu', dtype=torch.float32)
    buf521 = reinterpret_tensor(buf460, (8, 20, 28, 28), (15680, 784, 28, 1), 0); del buf460  # reuse
    cpp_fused_add_as_strided_scatter_convolution_backward_copy_native_batch_norm_backward_64(c_void_p(buf474.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_958.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf516.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf521.data_ptr()))
    del convolution_30
    del primals_92
    del squeeze_79
    del unsqueeze_958
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf522 = aten.convolution_backward(buf521, add_135, primals_91, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 20, [True, True, False])
    del add_135
    del primals_91
    buf523 = buf522[0]
    buf524 = buf522[1]
    del buf522
    buf525 = buf519; del buf519  # reuse
    buf526 = empty((20, ), device='cpu', dtype=torch.float32)
    buf527 = empty((20, ), device='cpu', dtype=torch.float32)
    buf528 = buf521; del buf521  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_65(c_void_p(buf516.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_970.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(buf528.data_ptr()))
    del buf523
    del convolution_29
    del primals_89
    del squeeze_76
    del unsqueeze_970
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf529 = aten.convolution_backward(buf528, mul_176, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_176
    del primals_88
    buf530 = buf529[0]
    buf531 = buf529[1]
    del buf529
    buf532 = reinterpret_tensor(buf332, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf332  # reuse
    buf533 = reinterpret_tensor(buf532, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf532  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_threshold_backward_66(c_void_p(buf533.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(cat_8.data_ptr()), c_void_p(bitwise_and_5.data_ptr()))
    del bitwise_and_5
    del cat_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.threshold_backward]
    buf534 = aten.convolution_backward(buf533, relu_12, primals_86, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf533
    del primals_86
    buf535 = buf534[0]
    buf536 = buf534[1]
    buf537 = buf534[2]
    del buf534
    buf538 = buf535; del buf535  # reuse
    cpp_fused_convolution_backward_threshold_backward_67(c_void_p(buf538.data_ptr()), c_void_p(relu_12.data_ptr()))
    del relu_12
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf539 = aten.convolution_backward(buf538, mean_1, primals_84, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf538
    del mean_1
    del primals_84
    buf540 = buf539[0]
    buf541 = buf539[1]
    buf542 = buf539[2]
    del buf539
    buf543 = empty((60, ), device='cpu', dtype=torch.float32)
    buf544 = empty((60, ), device='cpu', dtype=torch.float32)
    buf545 = reinterpret_tensor(buf493, (8, 60, 28, 28), (47040, 1, 1680, 60), 0); del buf493  # reuse
    buf546 = buf544; del buf544  # reuse
    buf547 = buf545; del buf545  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_68(c_void_p(buf546.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(le_30.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(div_1.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_982.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(buf543.data_ptr()))
    del convolution_26
    del le_30
    del primals_82
    del squeeze_73
    del unsqueeze_982
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf548 = aten.convolution_backward(buf547, relu_10, primals_81, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 60, [True, True, False])
    del buf547
    del primals_81
    buf549 = buf548[0]
    buf550 = buf548[1]
    del buf548
    buf551 = buf549; del buf549  # reuse
    buf552 = empty((60, ), device='cpu', dtype=torch.float32)
    buf553 = empty((60, ), device='cpu', dtype=torch.float32)
    buf554 = empty((60, ), device='cpu', dtype=torch.float32)
    buf555 = buf551; del buf551  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_69(c_void_p(buf555.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(div_1.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_994.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf554.data_ptr()))
    del buf530
    del buf540
    del buf553
    del convolution_25
    del div_1
    del primals_79
    del relu_10
    del squeeze_70
    del unsqueeze_994
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf556 = aten.convolution_backward(buf555, slice_44, primals_78, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf555
    del primals_78
    del slice_44
    buf557 = buf556[0]
    buf558 = buf556[1]
    del buf556
    buf559 = buf516; del buf516  # reuse
    buf560 = reinterpret_tensor(buf512, (250880, ), (1, ), 0); del buf512  # reuse
    buf563 = reinterpret_tensor(buf509, (8, 40, 28, 28), (31360, 784, 28, 1), 0); del buf509  # reuse
    buf565 = buf484; del buf484  # reuse
    buf566 = empty((40, ), device='cpu', dtype=torch.float32)
    buf567 = empty((40, ), device='cpu', dtype=torch.float32)
    buf568 = reinterpret_tensor(buf474, (8, 40, 28, 28), (31360, 784, 28, 1), 0); del buf474  # reuse
    cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_70(c_void_p(buf513.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_1006.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf568.data_ptr()))
    del buf513
    del buf557
    del buf559
    del buf560
    del buf566
    del convolution_24
    del primals_76
    del squeeze_67
    del unsqueeze_1006
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf569 = aten.convolution_backward(buf568, add_113, primals_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_113
    del buf568
    del primals_75
    buf570 = buf569[0]
    buf571 = buf569[1]
    del buf569
    buf572 = empty((24, ), device='cpu', dtype=torch.float32)
    buf573 = empty((24, ), device='cpu', dtype=torch.float32)
    buf574 = empty((24, ), device='cpu', dtype=torch.float32)
    buf575 = buf570; del buf570  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_71(c_void_p(buf575.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_1018.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(buf573.data_ptr()), c_void_p(buf574.data_ptr()))
    del convolution_23
    del primals_73
    del squeeze_64
    del unsqueeze_1018
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf576 = aten.convolution_backward(buf575, slice_33, primals_72, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 24, [True, True, False])
    del buf575
    del primals_72
    buf577 = buf576[0]
    buf578 = buf576[1]
    del buf576
    buf579 = buf526; del buf526  # reuse
    buf580 = empty((20, ), device='cpu', dtype=torch.float32)
    buf581 = empty((20, ), device='cpu', dtype=torch.float32)
    buf582 = buf528; del buf528  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_72(c_void_p(buf563.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(unsqueeze_1030.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf582.data_ptr()))
    del convolution_22
    del primals_70
    del squeeze_61
    del unsqueeze_1030
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf583 = aten.convolution_backward(buf582, add_103, primals_69, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 20, [True, True, False])
    del add_103
    del primals_69
    buf584 = buf583[0]
    buf585 = buf583[1]
    del buf583
    buf586 = buf580; del buf580  # reuse
    buf587 = empty((20, ), device='cpu', dtype=torch.float32)
    buf588 = empty((20, ), device='cpu', dtype=torch.float32)
    buf589 = buf582; del buf582  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_73(c_void_p(buf563.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_1042.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(buf589.data_ptr()))
    del buf563
    del buf584
    del buf587
    del convolution_21
    del primals_67
    del squeeze_58
    del unsqueeze_1042
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf590 = aten.convolution_backward(buf589, mul_133, primals_66, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf589
    del mul_133
    del primals_66
    buf591 = buf590[0]
    buf592 = buf590[1]
    del buf590
    buf593 = empty_strided((8, 72, 1, 1), (72, 1, 576, 576), device='cpu', dtype=torch.float32)
    buf594 = reinterpret_tensor(buf593, (8, 72, 1, 1), (72, 1, 72, 72), 0); del buf593  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_mul_sum_threshold_backward_74(c_void_p(buf594.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(add_97.data_ptr()), c_void_p(bitwise_and_6.data_ptr()))
    del add_97
    del bitwise_and_6
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.threshold_backward]
    buf595 = aten.convolution_backward(buf594, relu_9, primals_64, [72], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf594
    del primals_64
    buf596 = buf595[0]
    buf597 = buf595[1]
    buf598 = buf595[2]
    del buf595
    buf599 = buf596; del buf596  # reuse
    cpp_fused_convolution_backward_threshold_backward_75(c_void_p(buf599.data_ptr()), c_void_p(relu_9.data_ptr()))
    del relu_9
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf600 = aten.convolution_backward(buf599, mean, primals_62, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf599
    del mean
    del primals_62
    buf601 = buf600[0]
    buf602 = buf600[1]
    buf603 = buf600[2]
    del buf600
    buf604 = empty((72, ), device='cpu', dtype=torch.float32)
    buf605 = empty((72, ), device='cpu', dtype=torch.float32)
    buf606 = buf591; del buf591  # reuse
    buf607 = buf605; del buf605  # reuse
    cpp_fused_add_div_mul_native_batch_norm_backward_76(c_void_p(buf606.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(div.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_1054.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf604.data_ptr()))
    del buf601
    del convolution_18
    del div
    del primals_60
    del squeeze_55
    del unsqueeze_1054
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf608 = aten.convolution_backward(buf606, slice_36, primals_59, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 72, [True, True, False])
    del buf606
    del primals_59
    del slice_36
    buf609 = buf608[0]
    buf610 = buf608[1]
    del buf608
    buf611 = empty((36, ), device='cpu', dtype=torch.float32)
    buf612 = empty((36, ), device='cpu', dtype=torch.float32)
    buf613 = empty((36, ), device='cpu', dtype=torch.float32)
    buf614 = empty_strided((8, 36, 56, 56), (112896, 1, 2016, 36), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_77(c_void_p(le_33.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(unsqueeze_1066.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf614.data_ptr()))
    del convolution_17
    del le_33
    del primals_57
    del squeeze_52
    del unsqueeze_1066
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf615 = aten.convolution_backward(buf614, relu_7, primals_56, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 36, [True, True, False])
    del buf614
    del primals_56
    buf616 = buf615[0]
    buf617 = buf615[1]
    del buf615
    buf618 = buf612; del buf612  # reuse
    buf619 = empty((36, ), device='cpu', dtype=torch.float32)
    buf620 = buf616; del buf616  # reuse
    buf621 = buf619; del buf619  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_78(c_void_p(buf620.data_ptr()), c_void_p(buf621.data_ptr()), c_void_p(relu_7.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_1078.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf618.data_ptr()))
    del buf609
    del convolution_16
    del primals_54
    del relu_7
    del squeeze_49
    del unsqueeze_1078
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf622 = aten.convolution_backward(buf620, slice_33, primals_53, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_53
    del slice_33
    buf623 = buf622[0]
    buf624 = buf622[1]
    del buf622
    buf626 = empty((8, 24, 56, 56), device='cpu', dtype=torch.float32)
    buf627 = empty((602112, ), device='cpu', dtype=torch.float32)
    buf630 = empty((8, 24, 56, 56), device='cpu', dtype=torch.float32)
    buf632 = empty((12, ), device='cpu', dtype=torch.float32)
    buf633 = empty((12, ), device='cpu', dtype=torch.float32)
    buf634 = empty((12, ), device='cpu', dtype=torch.float32)
    buf635 = empty((8, 12, 56, 56), device='cpu', dtype=torch.float32)
    cpp_fused_add_as_strided_scatter_convolution_backward_copy_native_batch_norm_backward_79(c_void_p(buf577.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_1090.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf626.data_ptr()), c_void_p(buf627.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(buf633.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(buf635.data_ptr()))
    del convolution_15
    del primals_51
    del squeeze_46
    del unsqueeze_1090
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf636 = aten.convolution_backward(buf635, add_76, primals_50, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 12, [True, True, False])
    del add_76
    del primals_50
    buf637 = buf636[0]
    buf638 = buf636[1]
    del buf636
    buf639 = buf633; del buf633  # reuse
    buf640 = empty((12, ), device='cpu', dtype=torch.float32)
    buf641 = empty((12, ), device='cpu', dtype=torch.float32)
    buf642 = buf635; del buf635  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_80(c_void_p(buf630.data_ptr()), c_void_p(buf637.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_1102.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf639.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(buf641.data_ptr()), c_void_p(buf642.data_ptr()))
    del buf637
    del convolution_14
    del primals_48
    del squeeze_43
    del unsqueeze_1102
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf643 = aten.convolution_backward(buf642, slice_25, primals_47, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_47
    del slice_25
    buf644 = buf643[0]
    buf645 = buf643[1]
    del buf643
    buf646 = empty((36, ), device='cpu', dtype=torch.float32)
    buf647 = empty((36, ), device='cpu', dtype=torch.float32)
    buf648 = empty((36, ), device='cpu', dtype=torch.float32)
    buf649 = buf620; del buf620  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_81(c_void_p(le_35.data_ptr()), c_void_p(buf644.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_1114.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf647.data_ptr()), c_void_p(buf648.data_ptr()), c_void_p(buf649.data_ptr()))
    del convolution_13
    del le_35
    del primals_45
    del squeeze_40
    del unsqueeze_1114
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf650 = aten.convolution_backward(buf649, relu_5, primals_44, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 36, [True, True, False])
    del buf649
    del primals_44
    buf651 = buf650[0]
    buf652 = buf650[1]
    del buf650
    buf653 = buf647; del buf647  # reuse
    buf654 = empty((36, ), device='cpu', dtype=torch.float32)
    buf655 = buf651; del buf651  # reuse
    buf656 = buf654; del buf654  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_82(c_void_p(buf655.data_ptr()), c_void_p(buf656.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(buf644.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_1126.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf653.data_ptr()))
    del buf644
    del convolution_12
    del primals_42
    del relu_5
    del squeeze_37
    del unsqueeze_1126
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf657 = aten.convolution_backward(buf655, slice_22, primals_41, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf655
    del primals_41
    del slice_22
    buf658 = buf657[0]
    buf659 = buf657[1]
    del buf657
    buf660 = buf630; del buf630  # reuse
    buf661 = reinterpret_tensor(buf626, (602112, ), (1, ), 0); del buf626  # reuse
    buf664 = reinterpret_tensor(buf623, (8, 24, 56, 56), (75264, 3136, 56, 1), 0); del buf623  # reuse
    buf666 = buf573; del buf573  # reuse
    buf667 = empty((24, ), device='cpu', dtype=torch.float32)
    buf668 = empty((24, ), device='cpu', dtype=torch.float32)
    buf669 = reinterpret_tensor(buf577, (8, 24, 56, 56), (75264, 3136, 56, 1), 0); del buf577  # reuse
    cpp_fused_add_as_strided_scatter_convolution_backward_native_batch_norm_backward_83(c_void_p(buf627.data_ptr()), c_void_p(buf658.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_1138.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf660.data_ptr()), c_void_p(buf661.data_ptr()), c_void_p(buf664.data_ptr()), c_void_p(buf666.data_ptr()), c_void_p(buf667.data_ptr()), c_void_p(buf668.data_ptr()), c_void_p(buf669.data_ptr()))
    del buf627
    del buf658
    del buf660
    del buf661
    del convolution_11
    del primals_39
    del squeeze_34
    del unsqueeze_1138
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf670 = aten.convolution_backward(buf669, add_55, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_55
    del buf669
    del primals_38
    buf671 = buf670[0]
    buf672 = buf670[1]
    del buf670
    buf673 = empty((16, ), device='cpu', dtype=torch.float32)
    buf674 = empty((16, ), device='cpu', dtype=torch.float32)
    buf675 = empty((16, ), device='cpu', dtype=torch.float32)
    buf676 = buf671; del buf671  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_84(c_void_p(buf676.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_1150.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(buf674.data_ptr()), c_void_p(buf675.data_ptr()))
    del convolution_10
    del primals_36
    del squeeze_31
    del unsqueeze_1150
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf677 = aten.convolution_backward(buf676, slice_11, primals_35, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False])
    del buf676
    del primals_35
    buf678 = buf677[0]
    buf679 = buf677[1]
    del buf677
    buf680 = buf640; del buf640  # reuse
    buf681 = empty((12, ), device='cpu', dtype=torch.float32)
    buf682 = empty((12, ), device='cpu', dtype=torch.float32)
    buf683 = buf642; del buf642  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_85(c_void_p(buf664.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_1162.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf680.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(buf682.data_ptr()), c_void_p(buf683.data_ptr()))
    del convolution_9
    del primals_33
    del squeeze_28
    del unsqueeze_1162
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf684 = aten.convolution_backward(buf683, add_45, primals_32, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 12, [True, True, False])
    del add_45
    del primals_32
    buf685 = buf684[0]
    buf686 = buf684[1]
    del buf684
    buf687 = buf681; del buf681  # reuse
    buf688 = empty((12, ), device='cpu', dtype=torch.float32)
    buf689 = empty((12, ), device='cpu', dtype=torch.float32)
    buf690 = buf683; del buf683  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_86(c_void_p(buf664.data_ptr()), c_void_p(buf685.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_1174.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(buf688.data_ptr()), c_void_p(buf689.data_ptr()), c_void_p(buf690.data_ptr()))
    del buf664
    del buf685
    del buf688
    del convolution_8
    del primals_30
    del squeeze_25
    del unsqueeze_1174
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf691 = aten.convolution_backward(buf690, add_40, primals_29, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_40
    del buf690
    del primals_29
    buf692 = buf691[0]
    buf693 = buf691[1]
    del buf691
    buf694 = empty((48, ), device='cpu', dtype=torch.float32)
    buf695 = empty((48, ), device='cpu', dtype=torch.float32)
    buf696 = empty((48, ), device='cpu', dtype=torch.float32)
    buf697 = buf692; del buf692  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_87(c_void_p(buf697.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_1186.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf694.data_ptr()), c_void_p(buf695.data_ptr()), c_void_p(buf696.data_ptr()))
    del buf695
    del convolution_7
    del primals_27
    del squeeze_22
    del unsqueeze_1186
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf698 = aten.convolution_backward(buf697, slice_14, primals_26, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 48, [True, True, False])
    del buf697
    del primals_26
    del slice_14
    buf699 = buf698[0]
    buf700 = buf698[1]
    del buf698
    buf701 = buf667; del buf667  # reuse
    buf702 = empty((24, ), device='cpu', dtype=torch.float32)
    buf703 = empty((24, ), device='cpu', dtype=torch.float32)
    buf704 = empty_strided((8, 24, 112, 112), (301056, 1, 2688, 24), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_88(c_void_p(le_37.data_ptr()), c_void_p(buf699.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_1198.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf702.data_ptr()), c_void_p(buf703.data_ptr()), c_void_p(buf704.data_ptr()))
    del convolution_6
    del le_37
    del primals_24
    del squeeze_19
    del unsqueeze_1198
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf705 = aten.convolution_backward(buf704, relu_3, primals_23, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 24, [True, True, False])
    del buf704
    del primals_23
    buf706 = buf705[0]
    buf707 = buf705[1]
    del buf705
    buf708 = buf702; del buf702  # reuse
    buf709 = empty((24, ), device='cpu', dtype=torch.float32)
    buf710 = buf706; del buf706  # reuse
    buf711 = buf709; del buf709  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_89(c_void_p(buf710.data_ptr()), c_void_p(buf711.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(buf699.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_1210.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf708.data_ptr()))
    del buf699
    del convolution_5
    del primals_21
    del relu_3
    del squeeze_16
    del unsqueeze_1210
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf712 = aten.convolution_backward(buf710, slice_11, primals_20, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf710
    del primals_20
    del slice_11
    buf713 = buf712[0]
    buf714 = buf712[1]
    del buf712
    buf716 = empty((8, 16, 112, 112), device='cpu', dtype=torch.float32)
    buf717 = empty((1605632, ), device='cpu', dtype=torch.float32)
    buf720 = empty((8, 16, 112, 112), device='cpu', dtype=torch.float32)
    buf722 = empty((8, ), device='cpu', dtype=torch.float32)
    buf723 = empty((8, ), device='cpu', dtype=torch.float32)
    buf724 = empty((8, ), device='cpu', dtype=torch.float32)
    buf725 = empty((8, 8, 112, 112), device='cpu', dtype=torch.float32)
    cpp_fused_add_as_strided_scatter_convolution_backward_copy_native_batch_norm_backward_90(c_void_p(buf678.data_ptr()), c_void_p(buf713.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_1222.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf716.data_ptr()), c_void_p(buf717.data_ptr()), c_void_p(buf720.data_ptr()), c_void_p(buf722.data_ptr()), c_void_p(buf723.data_ptr()), c_void_p(buf724.data_ptr()), c_void_p(buf725.data_ptr()))
    del buf678
    del buf713
    del buf716
    del convolution_4
    del primals_18
    del squeeze_13
    del unsqueeze_1222
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf726 = aten.convolution_backward(buf725, add_19, primals_17, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del add_19
    del primals_17
    buf727 = buf726[0]
    buf728 = buf726[1]
    del buf726
    buf729 = buf723; del buf723  # reuse
    buf730 = empty((8, ), device='cpu', dtype=torch.float32)
    buf731 = empty((8, ), device='cpu', dtype=torch.float32)
    buf732 = buf725; del buf725  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_91(c_void_p(buf720.data_ptr()), c_void_p(buf727.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_1234.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf729.data_ptr()), c_void_p(buf730.data_ptr()), c_void_p(buf731.data_ptr()), c_void_p(buf732.data_ptr()))
    del buf720
    del buf727
    del convolution_3
    del primals_15
    del squeeze_10
    del unsqueeze_1234
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf733 = aten.convolution_backward(buf732, slice_3, primals_14, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_14
    del slice_3
    buf734 = buf733[0]
    buf735 = buf733[1]
    del buf733
    buf736 = buf730; del buf730  # reuse
    buf737 = empty((8, ), device='cpu', dtype=torch.float32)
    buf738 = empty((8, ), device='cpu', dtype=torch.float32)
    buf739 = reinterpret_tensor(buf732, (8, 8, 112, 112), (100352, 1, 896, 8), 0); del buf732  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_92(c_void_p(le_39.data_ptr()), c_void_p(buf734.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_1246.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf736.data_ptr()), c_void_p(buf737.data_ptr()), c_void_p(buf738.data_ptr()), c_void_p(buf739.data_ptr()))
    del convolution_2
    del le_39
    del primals_12
    del squeeze_7
    del unsqueeze_1246
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf740 = aten.convolution_backward(buf739, relu_1, primals_11, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf739
    del primals_11
    buf741 = buf740[0]
    buf742 = buf740[1]
    del buf740
    buf743 = buf737; del buf737  # reuse
    buf744 = empty((8, ), device='cpu', dtype=torch.float32)
    buf745 = buf741; del buf741  # reuse
    buf746 = buf744; del buf744  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_93(c_void_p(buf745.data_ptr()), c_void_p(buf746.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(buf734.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_1258.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf743.data_ptr()))
    del buf734
    del convolution_1
    del primals_9
    del relu_1
    del squeeze_4
    del unsqueeze_1258
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf747 = aten.convolution_backward(buf745, relu, primals_8, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf745
    del primals_8
    buf748 = buf747[0]
    buf749 = buf747[1]
    del buf747
    buf750 = buf674; del buf674  # reuse
    buf751 = empty((16, ), device='cpu', dtype=torch.float32)
    buf752 = buf748; del buf748  # reuse
    buf753 = buf751; del buf751  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_94(c_void_p(buf752.data_ptr()), c_void_p(buf753.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(buf717.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_1270.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf750.data_ptr()))
    del buf717
    del convolution
    del primals_6
    del relu
    del squeeze_1
    del unsqueeze_1270
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf754 = aten.convolution_backward(buf752, primals_513, primals_5, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf752
    del primals_5
    del primals_513
    buf755 = buf754[1]
    return (buf10, buf8, reinterpret_tensor(buf1, (1000, 1280), (1280, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), buf755, buf753, buf750, buf749, buf746, buf743, buf742, buf738, buf736, buf735, buf731, buf729, buf728, buf724, buf722, buf714, buf711, buf708, buf707, buf703, buf701, buf700, buf696, buf694, buf693, buf689, buf687, buf686, buf682, buf680, buf679, buf675, buf673, buf672, buf668, buf666, buf659, buf656, buf653, buf652, buf648, buf646, buf645, buf641, buf639, buf638, buf634, buf632, buf624, buf621, buf618, buf617, buf613, buf611, buf610, buf607, buf604, buf602, buf603, buf597, buf598, buf592, buf588, buf586, buf585, buf581, buf579, buf578, buf574, buf572, buf571, buf567, buf565, buf558, buf554, buf552, buf550, buf546, buf543, buf541, buf542, buf536, buf537, buf531, buf527, buf525, buf524, buf520, buf518, buf510, buf507, buf504, buf503, buf499, buf497, buf496, buf492, buf490, buf489, buf485, buf483, buf482, buf478, buf476, buf475, buf471, buf469, buf468, buf464, buf462, buf455, buf452, buf449, buf448, buf444, buf442, buf441, buf437, buf435, buf434, buf430, buf428, buf421, buf418, buf415, buf414, buf410, buf408, buf407, buf403, buf401, buf400, buf396, buf394, buf387, buf384, buf381, buf380, buf376, buf374, buf373, buf369, buf367, buf366, buf362, buf360, buf352, buf348, buf346, buf344, buf340, buf337, buf335, buf336, buf330, buf331, buf325, buf321, buf319, buf318, buf314, buf312, buf311, buf307, buf305, buf304, buf300, buf298, buf291, buf287, buf285, buf283, buf279, buf276, buf274, buf275, buf269, buf270, buf264, buf260, buf258, buf257, buf253, buf251, buf243, buf240, buf237, buf236, buf232, buf230, buf229, buf226, buf223, buf221, buf222, buf216, buf217, buf211, buf207, buf205, buf204, buf200, buf198, buf197, buf193, buf191, buf190, buf186, buf184, buf177, buf174, buf171, buf170, buf166, buf164, buf163, buf159, buf157, buf156, buf152, buf150, buf143, buf139, buf137, buf135, buf131, buf128, buf126, buf127, buf121, buf122, buf116, buf112, buf110, buf109, buf105, buf103, buf96, buf93, buf90, buf89, buf85, buf83, buf82, buf78, buf76, buf75, buf71, buf69, buf62, buf58, buf56, buf54, buf50, buf47, buf45, buf46, buf40, buf41, buf35, buf31, buf29, buf28, buf24, buf22, buf14, buf6, buf7, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((24, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((48, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((12, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((12, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((24, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((36, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((36, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((12, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((12, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((36, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((36, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((20, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((72, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((20, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((20, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((24, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((40, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((60, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((20, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((20, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((80, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((100, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((100, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((40, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((92, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((92, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((40, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((92, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((92, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((40, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((120, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((480, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((56, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((56, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((112, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((336, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((336, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((56, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((56, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((336, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((336, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((80, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((112, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((160, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((1280, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_513 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    relu = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    slice_3 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    add_19 = rand_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    slice_11 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 24, 112, 112), (301056, 1, 2688, 24), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((8, 24, 112, 112), (301056, 1, 2688, 24), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 24, 112, 112), (301056, 1, 2688, 24), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    slice_14 = rand_strided((8, 48, 112, 112), (602112, 1, 5376, 48), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    add_40 = rand_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 12, 56, 56), (37632, 1, 672, 12), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    add_45 = rand_strided((8, 12, 56, 56), (37632, 1, 672, 12), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 12, 56, 56), (37632, 1, 672, 12), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 16, 56, 56), (50176, 1, 896, 16), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    add_55 = rand_strided((8, 16, 56, 56), (50176, 1, 896, 16), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    slice_22 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 36, 56, 56), (112896, 1, 2016, 36), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((8, 36, 56, 56), (112896, 1, 2016, 36), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 36, 56, 56), (112896, 1, 2016, 36), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    slice_25 = rand_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 12, 56, 56), (37632, 1, 672, 12), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    add_76 = rand_strided((8, 12, 56, 56), (37632, 1, 672, 12), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 12, 56, 56), (37632, 1, 672, 12), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    slice_33 = rand_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 36, 56, 56), (112896, 1, 2016, 36), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((8, 36, 56, 56), (112896, 1, 2016, 36), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 36, 56, 56), (112896, 1, 2016, 36), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    slice_36 = rand_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    add_97 = rand_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cpu', dtype=torch.float32)
    mean = rand_strided((8, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    div = rand_strided((8, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    mul_133 = rand_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 20, 28, 28), (15680, 1, 560, 20), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    add_103 = rand_strided((8, 20, 28, 28), (15680, 1, 560, 20), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 20, 28, 28), (15680, 1, 560, 20), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 24, 28, 28), (18816, 1, 672, 24), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    add_113 = rand_strided((8, 24, 28, 28), (18816, 1, 672, 24), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    slice_44 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 60, 28, 28), (47040, 1, 1680, 60), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((8, 60, 28, 28), (47040, 1, 1680, 60), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 60, 28, 28), (47040, 1, 1680, 60), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    cat_8 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    mean_1 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((8, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    div_1 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    mul_176 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 20, 28, 28), (15680, 1, 560, 20), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    add_135 = rand_strided((8, 20, 28, 28), (15680, 1, 560, 20), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 20, 28, 28), (15680, 1, 560, 20), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    slice_55 = rand_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    slice_58 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    add_156 = rand_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    add_161 = rand_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    add_171 = rand_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    slice_66 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 100, 14, 14), (19600, 1, 1400, 100), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((8, 100, 14, 14), (19600, 1, 1400, 100), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((8, 100, 14, 14), (19600, 1, 1400, 100), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    slice_69 = rand_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    add_192 = rand_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    slice_77 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((8, 92, 14, 14), (18032, 1, 1288, 92), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    relu_17 = rand_strided((8, 92, 14, 14), (18032, 1, 1288, 92), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((8, 92, 14, 14), (18032, 1, 1288, 92), device='cpu', dtype=torch.float32)
    squeeze_118 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    slice_80 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cpu', dtype=torch.float32)
    squeeze_121 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    add_213 = rand_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    slice_88 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((8, 92, 14, 14), (18032, 1, 1288, 92), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    relu_19 = rand_strided((8, 92, 14, 14), (18032, 1, 1288, 92), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((8, 92, 14, 14), (18032, 1, 1288, 92), device='cpu', dtype=torch.float32)
    squeeze_130 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    slice_91 = rand_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cpu', dtype=torch.float32)
    squeeze_133 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    add_234 = rand_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cpu', dtype=torch.float32)
    squeeze_136 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    slice_99 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    squeeze_139 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    relu_21 = rand_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    squeeze_142 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    cat_18 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    mean_2 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    relu_23 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    div_2 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    mul_338 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_54 = rand_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cpu', dtype=torch.float32)
    squeeze_145 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    add_256 = rand_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cpu', dtype=torch.float32)
    squeeze_148 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_56 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    squeeze_151 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    add_266 = rand_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_57 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    squeeze_154 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    slice_110 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    convolution_58 = rand_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cpu', dtype=torch.float32)
    squeeze_157 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    relu_24 = rand_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cpu', dtype=torch.float32)
    convolution_59 = rand_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cpu', dtype=torch.float32)
    squeeze_160 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    cat_20 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    mean_3 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    relu_26 = rand_strided((8, 168, 1, 1), (168, 1, 168, 168), device='cpu', dtype=torch.float32)
    div_3 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    mul_381 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    convolution_62 = rand_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cpu', dtype=torch.float32)
    squeeze_163 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    add_288 = rand_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cpu', dtype=torch.float32)
    convolution_63 = rand_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cpu', dtype=torch.float32)
    squeeze_166 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    slice_121 = rand_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    convolution_64 = rand_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cpu', dtype=torch.float32)
    squeeze_169 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    relu_27 = rand_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cpu', dtype=torch.float32)
    convolution_65 = rand_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cpu', dtype=torch.float32)
    squeeze_172 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    slice_124 = rand_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    convolution_66 = rand_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    squeeze_175 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    add_309 = rand_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    mean_4 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    relu_29 = rand_strided((8, 168, 1, 1), (168, 1, 168, 168), device='cpu', dtype=torch.float32)
    div_4 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    mul_417 = rand_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    convolution_69 = rand_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    squeeze_178 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    add_315 = rand_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    convolution_70 = rand_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    squeeze_181 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_71 = rand_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cpu', dtype=torch.float32)
    squeeze_184 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    add_325 = rand_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cpu', dtype=torch.float32)
    convolution_72 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    squeeze_187 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    slice_132 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    convolution_73 = rand_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.float32)
    squeeze_190 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    relu_30 = rand_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.float32)
    convolution_74 = rand_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.float32)
    squeeze_193 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    slice_135 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    convolution_75 = rand_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    squeeze_196 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    add_346 = rand_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    convolution_76 = rand_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    squeeze_199 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    slice_143 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    convolution_77 = rand_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.float32)
    squeeze_202 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    relu_32 = rand_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.float32)
    convolution_78 = rand_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.float32)
    squeeze_205 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    cat_26 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    mean_5 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    relu_34 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    div_5 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    mul_488 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    convolution_81 = rand_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    squeeze_208 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    add_368 = rand_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    convolution_82 = rand_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    squeeze_211 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    slice_154 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    convolution_83 = rand_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.float32)
    squeeze_214 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    relu_35 = rand_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.float32)
    convolution_84 = rand_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.float32)
    squeeze_217 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    slice_157 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    convolution_85 = rand_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    squeeze_220 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    add_389 = rand_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    convolution_86 = rand_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    squeeze_223 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    slice_165 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    convolution_87 = rand_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.float32)
    squeeze_226 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    relu_37 = rand_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.float32)
    convolution_88 = rand_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.float32)
    squeeze_229 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    cat_30 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    mean_6 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    relu_39 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    div_6 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    mul_545 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    convolution_91 = rand_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    squeeze_232 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    add_411 = rand_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    convolution_92 = rand_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    squeeze_235 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    slice_176 = rand_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    convolution_93 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    squeeze_238 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    mean_7 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    view_1 = rand_strided((8, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((8, 1280, 1, 1), (1280, 1, 1280, 1280), device='cpu', dtype=torch.bool)
    le_1 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.bool)
    unsqueeze_322 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.bool)
    le_3 = rand_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.bool)
    unsqueeze_358 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_370 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_382 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_5 = rand_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.bool)
    unsqueeze_406 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_418 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_430 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_1 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.bool)
    le_8 = rand_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.bool)
    unsqueeze_454 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_466 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_478 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_10 = rand_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.bool)
    unsqueeze_502 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_514 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_526 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_538 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_550 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_562 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_2 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.bool)
    unsqueeze_574 = rand_strided((1, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_13 = rand_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cpu', dtype=torch.bool)
    unsqueeze_586 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_598 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_610 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_622 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_3 = rand_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.bool)
    le_16 = rand_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cpu', dtype=torch.bool)
    unsqueeze_634 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_646 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_658 = rand_strided((1, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_670 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_682 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_694 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_4 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.bool)
    le_19 = rand_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.bool)
    unsqueeze_706 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_718 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_730 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_742 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_21 = rand_strided((8, 92, 14, 14), (18032, 1, 1288, 92), device='cpu', dtype=torch.bool)
    unsqueeze_754 = rand_strided((1, 92, 1, 1), (92, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_766 = rand_strided((1, 92, 1, 1), (92, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_778 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_790 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_23 = rand_strided((8, 92, 14, 14), (18032, 1, 1288, 92), device='cpu', dtype=torch.bool)
    unsqueeze_802 = rand_strided((1, 92, 1, 1), (92, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_814 = rand_strided((1, 92, 1, 1), (92, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_826 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_838 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_25 = rand_strided((8, 100, 14, 14), (19600, 1, 1400, 100), device='cpu', dtype=torch.bool)
    unsqueeze_850 = rand_strided((1, 100, 1, 1), (100, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_862 = rand_strided((1, 100, 1, 1), (100, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_874 = rand_strided((1, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_886 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_898 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_910 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_922 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_27 = rand_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.bool)
    unsqueeze_934 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_946 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_958 = rand_strided((1, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_970 = rand_strided((1, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_5 = rand_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.bool)
    le_30 = rand_strided((8, 60, 28, 28), (47040, 1, 1680, 60), device='cpu', dtype=torch.bool)
    unsqueeze_982 = rand_strided((1, 60, 1, 1), (60, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_994 = rand_strided((1, 60, 1, 1), (60, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1006 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1018 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1030 = rand_strided((1, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1042 = rand_strided((1, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_and_6 = rand_strided((8, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.bool)
    unsqueeze_1054 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_33 = rand_strided((8, 36, 56, 56), (112896, 1, 2016, 36), device='cpu', dtype=torch.bool)
    unsqueeze_1066 = rand_strided((1, 36, 1, 1), (36, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1078 = rand_strided((1, 36, 1, 1), (36, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1090 = rand_strided((1, 12, 1, 1), (12, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1102 = rand_strided((1, 12, 1, 1), (12, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_35 = rand_strided((8, 36, 56, 56), (112896, 1, 2016, 36), device='cpu', dtype=torch.bool)
    unsqueeze_1114 = rand_strided((1, 36, 1, 1), (36, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1126 = rand_strided((1, 36, 1, 1), (36, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1138 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1150 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1162 = rand_strided((1, 12, 1, 1), (12, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1174 = rand_strided((1, 12, 1, 1), (12, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1186 = rand_strided((1, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_37 = rand_strided((8, 24, 112, 112), (301056, 1, 2688, 24), device='cpu', dtype=torch.bool)
    unsqueeze_1198 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1210 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1222 = rand_strided((1, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1234 = rand_strided((1, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    le_39 = rand_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cpu', dtype=torch.bool)
    unsqueeze_1246 = rand_strided((1, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1258 = rand_strided((1, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1270 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_15, primals_17, primals_18, primals_20, primals_21, primals_23, primals_24, primals_26, primals_27, primals_29, primals_30, primals_32, primals_33, primals_35, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_47, primals_48, primals_50, primals_51, primals_53, primals_54, primals_56, primals_57, primals_59, primals_60, primals_62, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_159, primals_161, primals_162, primals_164, primals_165, primals_167, primals_168, primals_170, primals_171, primals_173, primals_174, primals_176, primals_177, primals_179, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_234, primals_236, primals_237, primals_239, primals_240, primals_242, primals_243, primals_245, primals_246, primals_248, primals_249, primals_251, primals_252, primals_254, primals_255, primals_257, primals_258, primals_260, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_513, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, slice_3, convolution_3, squeeze_10, add_19, convolution_4, squeeze_13, slice_11, convolution_5, squeeze_16, relu_3, convolution_6, squeeze_19, slice_14, convolution_7, squeeze_22, add_40, convolution_8, squeeze_25, add_45, convolution_9, squeeze_28, convolution_10, squeeze_31, add_55, convolution_11, squeeze_34, slice_22, convolution_12, squeeze_37, relu_5, convolution_13, squeeze_40, slice_25, convolution_14, squeeze_43, add_76, convolution_15, squeeze_46, slice_33, convolution_16, squeeze_49, relu_7, convolution_17, squeeze_52, slice_36, convolution_18, squeeze_55, add_97, mean, relu_9, div, mul_133, convolution_21, squeeze_58, add_103, convolution_22, squeeze_61, convolution_23, squeeze_64, add_113, convolution_24, squeeze_67, slice_44, convolution_25, squeeze_70, relu_10, convolution_26, squeeze_73, cat_8, mean_1, relu_12, div_1, mul_176, convolution_29, squeeze_76, add_135, convolution_30, squeeze_79, slice_55, convolution_31, squeeze_82, relu_13, convolution_32, squeeze_85, slice_58, convolution_33, squeeze_88, add_156, convolution_34, squeeze_91, add_161, convolution_35, squeeze_94, convolution_36, squeeze_97, add_171, convolution_37, squeeze_100, slice_66, convolution_38, squeeze_103, relu_15, convolution_39, squeeze_106, slice_69, convolution_40, squeeze_109, add_192, convolution_41, squeeze_112, slice_77, convolution_42, squeeze_115, relu_17, convolution_43, squeeze_118, slice_80, convolution_44, squeeze_121, add_213, convolution_45, squeeze_124, slice_88, convolution_46, squeeze_127, relu_19, convolution_47, squeeze_130, slice_91, convolution_48, squeeze_133, add_234, convolution_49, squeeze_136, slice_99, convolution_50, squeeze_139, relu_21, convolution_51, squeeze_142, cat_18, mean_2, relu_23, div_2, mul_338, convolution_54, squeeze_145, add_256, convolution_55, squeeze_148, convolution_56, squeeze_151, add_266, convolution_57, squeeze_154, slice_110, convolution_58, squeeze_157, relu_24, convolution_59, squeeze_160, cat_20, mean_3, relu_26, div_3, mul_381, convolution_62, squeeze_163, add_288, convolution_63, squeeze_166, slice_121, convolution_64, squeeze_169, relu_27, convolution_65, squeeze_172, slice_124, convolution_66, squeeze_175, add_309, mean_4, relu_29, div_4, mul_417, convolution_69, squeeze_178, add_315, convolution_70, squeeze_181, convolution_71, squeeze_184, add_325, convolution_72, squeeze_187, slice_132, convolution_73, squeeze_190, relu_30, convolution_74, squeeze_193, slice_135, convolution_75, squeeze_196, add_346, convolution_76, squeeze_199, slice_143, convolution_77, squeeze_202, relu_32, convolution_78, squeeze_205, cat_26, mean_5, relu_34, div_5, mul_488, convolution_81, squeeze_208, add_368, convolution_82, squeeze_211, slice_154, convolution_83, squeeze_214, relu_35, convolution_84, squeeze_217, slice_157, convolution_85, squeeze_220, add_389, convolution_86, squeeze_223, slice_165, convolution_87, squeeze_226, relu_37, convolution_88, squeeze_229, cat_30, mean_6, relu_39, div_6, mul_545, convolution_91, squeeze_232, add_411, convolution_92, squeeze_235, slice_176, convolution_93, squeeze_238, mean_7, view_1, permute_1, le, le_1, unsqueeze_322, unsqueeze_334, unsqueeze_346, bitwise_and, le_3, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, le_5, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, bitwise_and_1, le_8, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, le_10, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, bitwise_and_2, unsqueeze_574, le_13, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, bitwise_and_3, le_16, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, bitwise_and_4, le_19, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, le_21, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, le_23, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, le_25, unsqueeze_850, unsqueeze_862, unsqueeze_874, unsqueeze_886, unsqueeze_898, unsqueeze_910, unsqueeze_922, le_27, unsqueeze_934, unsqueeze_946, unsqueeze_958, unsqueeze_970, bitwise_and_5, le_30, unsqueeze_982, unsqueeze_994, unsqueeze_1006, unsqueeze_1018, unsqueeze_1030, unsqueeze_1042, bitwise_and_6, unsqueeze_1054, le_33, unsqueeze_1066, unsqueeze_1078, unsqueeze_1090, unsqueeze_1102, le_35, unsqueeze_1114, unsqueeze_1126, unsqueeze_1138, unsqueeze_1150, unsqueeze_1162, unsqueeze_1174, unsqueeze_1186, le_37, unsqueeze_1198, unsqueeze_1210, unsqueeze_1222, unsqueeze_1234, le_39, unsqueeze_1246, unsqueeze_1258, unsqueeze_1270, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('ghostnet_100', benchmark_compiled_module)
