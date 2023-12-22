
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


cpp_fused_add_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                       const float* in_ptr11,
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(301056L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_div_native_batch_norm_backward_sum_1 = async_compile.cpp('''
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
                       const float* in_ptr9,
                       const float* in_ptr10,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr5)
{
    auto out_ptr4 = in_out_ptr0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = tmp6 - tmp11;
                    tmp10.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    tmp12.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0 + (768L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0 + (768L*x2) + (37632L*x1)));
                            auto tmp1 = static_cast<float>(49.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = tmp3 * tmp4;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                            tmp_acc1_vec = tmp_acc1_vec + tmp5;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr1 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)), static_cast<long>(768L), tmp3, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = in_ptr8[static_cast<long>(x1 + x1_inner + (768L*x0))];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(tmp3 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = out_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp8 = in_ptr9[static_cast<long>(x1 + x1_inner)];
                            auto tmp15 = out_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp19 = in_ptr10[static_cast<long>(x1 + x1_inner)];
                            auto tmp1 = static_cast<float>(49.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp6 = static_cast<float>(0.002551020408163265);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                            auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp13 = at::vec::Vectorized<float>(tmp2);
                            auto tmp14 = tmp13 - tmp12;
                            auto tmp16 = decltype(tmp15)(tmp15 * tmp6);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp14 - tmp17;
                            auto tmp20 = decltype(tmp8)(tmp8 * tmp19);
                            auto tmp21 = at::vec::Vectorized<float>(tmp20);
                            auto tmp22 = tmp18 * tmp21;
                            tmp22.store(out_ptr5 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                        auto tmp1 = static_cast<float>(49.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        auto tmp6 = static_cast<float>(0.002551020408163265);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = tmp9 * tmp9;
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp4 * tmp11;
                        auto tmp13 = tmp3 - tmp12;
                        auto tmp15 = tmp14 * tmp7;
                        auto tmp16 = tmp13 - tmp15;
                        auto tmp18 = tmp9 * tmp17;
                        auto tmp19 = tmp16 * tmp18;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp19.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr5[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)), static_cast<long>(768L), tmp1, 8);
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)), static_cast<long>(768L), tmp3, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(tmp3 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp8 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                            auto tmp18 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = static_cast<float>(0.002551020408163265);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                            auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp13 = tmp2 - tmp12;
                            auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                            auto tmp16 = at::vec::Vectorized<float>(tmp15);
                            auto tmp17 = tmp13 - tmp16;
                            auto tmp19 = decltype(tmp8)(tmp8 * tmp18);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp17 * tmp20;
                            auto tmp22 = tmp0 + tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp4 = static_cast<float>(0.002551020408163265);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = tmp7 * tmp7;
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = tmp2 * tmp9;
                        auto tmp11 = tmp1 - tmp10;
                        auto tmp13 = tmp12 * tmp5;
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = tmp7 * tmp15;
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp18 = tmp0 + tmp17;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_bmm_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(49L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*(static_cast<long>(x1) % static_cast<long>(6L))) + (768L*x0) + (37632L*(c10::div_floor_integer(x1, 6L)))));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (6144L*x0)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (6144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (294L*x2) + (14406L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x3) + (294L*x2) + (14406L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (294L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (294L*x2) + (14406L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (294L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.08838834764831845);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x3) + (294L*x2) + (14406L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (294L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            auto tmp6 = static_cast<float>(0.08838834764831845);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0))] = tmp7;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_6 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x0 + (8L*(c10::div_floor_integer(x1, 768L))));
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(8);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((128L*x2) + (128L*x2_inner) + (6272L*(static_cast<long>(c10::div_floor_integer(x1, 128L)) % static_cast<long>(6L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))) + (static_cast<long>(x1) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(16);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-301056L) + x2 + (49L*(static_cast<long>(x1) % static_cast<long>(768L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L)))), to_float_mask(tmp11));
                            return tmp13;
                        }
                        ;
                        auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<int>(24);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((-602112L) + (128L*x2) + (128L*x2_inner) + (6272L*(static_cast<long>(c10::div_floor_integer(x1, 128L)) % static_cast<long>(6L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))) + (static_cast<long>(x1) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp15)); })();
                            return tmp19;
                        }
                        ;
                        auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                        auto tmp21 = to_float_mask(tmp11);
                        auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                        auto tmp23 = to_float_mask(tmp4);
                        auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                        tmp24.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (112896L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x0 + (8L*(c10::div_floor_integer(x1, 768L))));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(8);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((128L*x2) + (6272L*(static_cast<long>(c10::div_floor_integer(x1, 128L)) % static_cast<long>(6L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))) + (static_cast<long>(x1) % static_cast<long>(128L)))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(16);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-301056L) + x2 + (49L*(static_cast<long>(x1) % static_cast<long>(768L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(24);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr2[static_cast<long>((-602112L) + (128L*x2) + (6272L*(static_cast<long>(c10::div_floor_integer(x1, 128L)) % static_cast<long>(6L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))) + (static_cast<long>(x1) % static_cast<long>(128L)))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp11 ? tmp14 : tmp20;
                        auto tmp22 = tmp4 ? tmp7 : tmp21;
                        out_ptr0[static_cast<long>(x2 + (49L*x1) + (112896L*x0))] = tmp22;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_7 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp0 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(0.002551020408163265);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp0 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)), static_cast<long>(768L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(in_out_ptr2 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr2[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr2[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_9 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp0 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (49L*x2) + (37632L*x0)), static_cast<long>(49L), tmp0, 8);
                        float tmp24[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (37632L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (37632L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (37632L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = tmp5 - tmp6;
                            auto tmp9 = static_cast<float>(0.002551020408163265);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 * tmp10;
                            auto tmp13 = tmp12 * tmp12;
                            auto tmp14 = tmp11 * tmp13;
                            auto tmp15 = tmp7 * tmp14;
                            auto tmp16 = tmp2 - tmp15;
                            auto tmp18 = tmp17 * tmp10;
                            auto tmp19 = tmp16 - tmp18;
                            auto tmp21 = tmp12 * tmp20;
                            auto tmp22 = tmp19 * tmp21;
                            auto tmp23 = tmp1 + tmp22;
                            tmp23.store(tmp24 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp24, 8, in_out_ptr0 + static_cast<long>(x1 + (49L*x2) + (37632L*x0)), static_cast<long>(49L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x2) + (37632L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (37632L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (37632L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (37632L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp7 = out_ptr1[static_cast<long>(x2)];
                        auto tmp10 = in_ptr4[static_cast<long>(x2)];
                        auto tmp15 = out_ptr0[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(0.002551020408163265);
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp11 = decltype(tmp10)(tmp10 * tmp10);
                        auto tmp12 = decltype(tmp9)(tmp9 * tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        auto tmp14 = decltype(tmp1)(tmp1 - tmp13);
                        auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                        auto tmp17 = decltype(tmp14)(tmp14 - tmp16);
                        auto tmp19 = decltype(tmp10)(tmp10 * tmp18);
                        auto tmp20 = decltype(tmp17)(tmp17 * tmp19);
                        auto tmp21 = decltype(tmp0)(tmp0 + tmp20);
                        in_out_ptr0[static_cast<long>(x1 + (49L*x2) + (37632L*x0))] = tmp21;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_bmm_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(49L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*(static_cast<long>(x1) % static_cast<long>(6L))) + (768L*x0) + (37632L*(c10::div_floor_integer(x1, 6L)))));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (6144L*x0)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (6144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (294L*x2) + (14406L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x3) + (294L*x2) + (14406L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (294L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (294L*x2) + (14406L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (294L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.08838834764831845);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x3) + (294L*x2) + (14406L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (294L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            auto tmp6 = static_cast<float>(0.08838834764831845);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0))] = tmp7;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_12 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x0 + (8L*(c10::div_floor_integer(x1, 768L))));
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(8);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((128L*x2) + (128L*x2_inner) + (6272L*(static_cast<long>(c10::div_floor_integer(x1, 128L)) % static_cast<long>(6L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))) + (static_cast<long>(x1) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(16);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-301056L) + x2 + (49L*(static_cast<long>(x1) % static_cast<long>(768L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L)))), to_float_mask(tmp11));
                            return tmp13;
                        }
                        ;
                        auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<int>(24);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((-602112L) + (128L*x2) + (128L*x2_inner) + (6272L*(static_cast<long>(c10::div_floor_integer(x1, 128L)) % static_cast<long>(6L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))) + (static_cast<long>(x1) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp15)); })();
                            return tmp19;
                        }
                        ;
                        auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                        auto tmp21 = to_float_mask(tmp11);
                        auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                        auto tmp23 = to_float_mask(tmp4);
                        auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                        tmp24.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (112896L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x0 + (8L*(c10::div_floor_integer(x1, 768L))));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(8);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((128L*x2) + (6272L*(static_cast<long>(c10::div_floor_integer(x1, 128L)) % static_cast<long>(6L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))) + (static_cast<long>(x1) % static_cast<long>(128L)))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(16);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-301056L) + x2 + (49L*(static_cast<long>(x1) % static_cast<long>(768L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(24);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr2[static_cast<long>((-602112L) + (128L*x2) + (6272L*(static_cast<long>(c10::div_floor_integer(x1, 128L)) % static_cast<long>(6L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))) + (static_cast<long>(x1) % static_cast<long>(128L)))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp11 ? tmp14 : tmp20;
                        auto tmp22 = tmp4 ? tmp7 : tmp21;
                        out_ptr0[static_cast<long>(x2 + (49L*x1) + (112896L*x0))] = tmp22;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)), static_cast<long>(768L), tmp1, 8);
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)), static_cast<long>(768L), tmp3, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(tmp3 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp8 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp11 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = out_ptr0[static_cast<long>(x1 + x1_inner)];
                            auto tmp21 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 - tmp6;
                            auto tmp9 = static_cast<float>(0.002551020408163265);
                            auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                            auto tmp12 = decltype(tmp11)(tmp11 * tmp11);
                            auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp7 * tmp14;
                            auto tmp16 = tmp2 - tmp15;
                            auto tmp18 = decltype(tmp17)(tmp17 * tmp9);
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp16 - tmp19;
                            auto tmp22 = decltype(tmp11)(tmp11 * tmp21);
                            auto tmp23 = at::vec::Vectorized<float>(tmp22);
                            auto tmp24 = tmp20 * tmp23;
                            auto tmp25 = tmp0 + tmp24;
                            tmp25.store(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp6 = static_cast<float>(0.002551020408163265);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 * tmp7;
                        auto tmp10 = tmp9 * tmp9;
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp4 * tmp11;
                        auto tmp13 = tmp1 - tmp12;
                        auto tmp15 = tmp14 * tmp7;
                        auto tmp16 = tmp13 - tmp15;
                        auto tmp18 = tmp9 * tmp17;
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp0 + tmp19;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp20.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_15 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 - tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)), static_cast<long>(768L), tmp1, 8);
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)), static_cast<long>(768L), tmp3, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(tmp3 + static_cast<long>(8L*x1_inner));
                            auto tmp5 = out_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp8 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = out_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp18 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp6 = static_cast<float>(0.002551020408163265);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                            auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp13 = tmp2 - tmp12;
                            auto tmp15 = decltype(tmp14)(tmp14 * tmp6);
                            auto tmp16 = at::vec::Vectorized<float>(tmp15);
                            auto tmp17 = tmp13 - tmp16;
                            auto tmp19 = decltype(tmp8)(tmp8 * tmp18);
                            auto tmp20 = at::vec::Vectorized<float>(tmp19);
                            auto tmp21 = tmp17 * tmp20;
                            auto tmp22 = tmp0 + tmp21;
                            tmp22.store(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                        auto tmp4 = static_cast<float>(0.002551020408163265);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = tmp7 * tmp7;
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = tmp2 * tmp9;
                        auto tmp11 = tmp1 - tmp10;
                        auto tmp13 = tmp12 * tmp5;
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = tmp7 * tmp15;
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp18 = tmp0 + tmp17;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_bmm_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(49L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*(static_cast<long>(x1) % static_cast<long>(6L))) + (768L*x0) + (37632L*(c10::div_floor_integer(x1, 6L)))));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (6144L*x0)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (6144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (294L*x2) + (14406L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x3) + (294L*x2) + (14406L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (294L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (294L*x2) + (14406L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (294L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.08838834764831845);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x3) + (294L*x2) + (14406L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (294L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            auto tmp6 = static_cast<float>(0.08838834764831845);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0))] = tmp7;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_18 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x0 + (8L*(c10::div_floor_integer(x1, 768L))));
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(8);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((128L*x2) + (128L*x2_inner) + (6272L*(static_cast<long>(c10::div_floor_integer(x1, 128L)) % static_cast<long>(6L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))) + (static_cast<long>(x1) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(16);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-301056L) + x2 + (49L*(static_cast<long>(x1) % static_cast<long>(768L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L)))), to_float_mask(tmp11));
                            return tmp13;
                        }
                        ;
                        auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<int>(24);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((-602112L) + (128L*x2) + (128L*x2_inner) + (6272L*(static_cast<long>(c10::div_floor_integer(x1, 128L)) % static_cast<long>(6L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))) + (static_cast<long>(x1) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp15)); })();
                            return tmp19;
                        }
                        ;
                        auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                        auto tmp21 = to_float_mask(tmp11);
                        auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                        auto tmp23 = to_float_mask(tmp4);
                        auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                        tmp24.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (112896L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x0 + (8L*(c10::div_floor_integer(x1, 768L))));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(8);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((128L*x2) + (6272L*(static_cast<long>(c10::div_floor_integer(x1, 128L)) % static_cast<long>(6L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))) + (static_cast<long>(x1) % static_cast<long>(128L)))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(16);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-301056L) + x2 + (49L*(static_cast<long>(x1) % static_cast<long>(768L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(24);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr2[static_cast<long>((-602112L) + (128L*x2) + (6272L*(static_cast<long>(c10::div_floor_integer(x1, 128L)) % static_cast<long>(6L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))) + (static_cast<long>(x1) % static_cast<long>(128L)))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp11 ? tmp14 : tmp20;
                        auto tmp22 = tmp4 ? tmp7 : tmp21;
                        out_ptr0[static_cast<long>(x2 + (49L*x1) + (112896L*x0))] = tmp22;
                    }
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
                       float* in_out_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp0 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(0.002551020408163265);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp0 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)), static_cast<long>(768L), tmp1, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp0 + tmp2;
                            tmp3.store(in_out_ptr2 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr2[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) in_out_ptr2[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (37632L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_21 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp0 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (49L*x2) + (37632L*x0)), static_cast<long>(49L), tmp0, 8);
                        float tmp24[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (37632L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (37632L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (37632L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = tmp5 - tmp6;
                            auto tmp9 = static_cast<float>(0.002551020408163265);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 * tmp10;
                            auto tmp13 = tmp12 * tmp12;
                            auto tmp14 = tmp11 * tmp13;
                            auto tmp15 = tmp7 * tmp14;
                            auto tmp16 = tmp2 - tmp15;
                            auto tmp18 = tmp17 * tmp10;
                            auto tmp19 = tmp16 - tmp18;
                            auto tmp21 = tmp12 * tmp20;
                            auto tmp22 = tmp19 * tmp21;
                            auto tmp23 = tmp1 + tmp22;
                            tmp23.store(tmp24 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp24, 8, in_out_ptr0 + static_cast<long>(x1 + (49L*x2) + (37632L*x0)), static_cast<long>(49L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x2) + (37632L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (37632L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (37632L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (768L*x1) + (37632L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp7 = out_ptr1[static_cast<long>(x2)];
                        auto tmp10 = in_ptr4[static_cast<long>(x2)];
                        auto tmp15 = out_ptr0[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(0.002551020408163265);
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp11 = decltype(tmp10)(tmp10 * tmp10);
                        auto tmp12 = decltype(tmp9)(tmp9 * tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        auto tmp14 = decltype(tmp1)(tmp1 - tmp13);
                        auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                        auto tmp17 = decltype(tmp14)(tmp14 - tmp16);
                        auto tmp19 = decltype(tmp10)(tmp10 * tmp18);
                        auto tmp20 = decltype(tmp17)(tmp17 * tmp19);
                        auto tmp21 = decltype(tmp0)(tmp0 + tmp20);
                        in_out_ptr0[static_cast<long>(x1 + (49L*x2) + (37632L*x0))] = tmp21;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_bmm_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(49L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*(static_cast<long>(x1) % static_cast<long>(6L))) + (768L*x0) + (37632L*(c10::div_floor_integer(x1, 6L)))));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (6144L*x0)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (6144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (294L*x2) + (14406L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x3) + (294L*x2) + (14406L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (294L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (294L*x2) + (14406L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (294L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.08838834764831845);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x3) + (294L*x2) + (14406L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (294L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            auto tmp6 = static_cast<float>(0.08838834764831845);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (14406L*x0))] = tmp7;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_24 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x0 + (8L*(c10::div_floor_integer(x1, 768L))));
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(8);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((128L*x2) + (128L*x2_inner) + (6272L*(static_cast<long>(c10::div_floor_integer(x1, 128L)) % static_cast<long>(6L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))) + (static_cast<long>(x1) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(16);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-301056L) + x2 + (49L*(static_cast<long>(x1) % static_cast<long>(768L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L)))), to_float_mask(tmp11));
                            return tmp13;
                        }
                        ;
                        auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<int>(24);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((-602112L) + (128L*x2) + (128L*x2_inner) + (6272L*(static_cast<long>(c10::div_floor_integer(x1, 128L)) % static_cast<long>(6L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))) + (static_cast<long>(x1) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp15)); })();
                            return tmp19;
                        }
                        ;
                        auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                        auto tmp21 = to_float_mask(tmp11);
                        auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                        auto tmp23 = to_float_mask(tmp4);
                        auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                        tmp24.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (112896L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x0 + (8L*(c10::div_floor_integer(x1, 768L))));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(8);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((128L*x2) + (6272L*(static_cast<long>(c10::div_floor_integer(x1, 128L)) % static_cast<long>(6L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))) + (static_cast<long>(x1) % static_cast<long>(128L)))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(16);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-301056L) + x2 + (49L*(static_cast<long>(x1) % static_cast<long>(768L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(24);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr2[static_cast<long>((-602112L) + (128L*x2) + (6272L*(static_cast<long>(c10::div_floor_integer(x1, 128L)) % static_cast<long>(6L))) + (37632L*x0) + (301056L*(c10::div_floor_integer(x1, 768L))) + (static_cast<long>(x1) % static_cast<long>(128L)))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp11 ? tmp14 : tmp20;
                        auto tmp22 = tmp4 ? tmp7 : tmp21;
                        out_ptr0[static_cast<long>(x2 + (49L*x1) + (112896L*x0))] = tmp22;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_sum_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr4 + static_cast<long>(x1 + (49L*x2) + (37632L*x0)), static_cast<long>(49L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (37632L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (37632L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
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
                            auto tmp21 = tmp1 + tmp20;
                            tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (37632L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x1 + (49L*x2) + (37632L*x0))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (37632L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (768L*x1) + (37632L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2)];
                        auto tmp5 = out_ptr1[static_cast<long>(x2)];
                        auto tmp8 = in_ptr3[static_cast<long>(x2)];
                        auto tmp13 = out_ptr0[static_cast<long>(x2)];
                        auto tmp16 = in_ptr5[static_cast<long>(x2)];
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(0.002551020408163265);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp12 = decltype(tmp1)(tmp1 - tmp11);
                        auto tmp14 = decltype(tmp13)(tmp13 * tmp6);
                        auto tmp15 = decltype(tmp12)(tmp12 - tmp14);
                        auto tmp17 = decltype(tmp8)(tmp8 * tmp16);
                        auto tmp18 = decltype(tmp15)(tmp15 * tmp17);
                        auto tmp19 = decltype(tmp0)(tmp0 + tmp18);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (37632L*x0))] = tmp19;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>(x0 + (768L*x1) + (768L*x1_inner) + (37632L*x2))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (49L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x0 + (768L*x1) + (37632L*x2))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr3[static_cast<long>(x1 + (49L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr6 + static_cast<long>(x0));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        float tmp19[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (37632L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (768L*x1) + (768L*x1_inner) + (37632L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x2));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2));
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
                            tmp18.store(tmp19 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp19, 8, out_ptr7 + static_cast<long>(x1 + (49L*x2) + (37632L*x0)), static_cast<long>(49L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (37632L*x0))];
                        auto tmp1 = in_ptr6[static_cast<long>(x2 + (768L*x1) + (37632L*x0))];
                        auto tmp2 = in_ptr7[static_cast<long>(x2)];
                        auto tmp4 = out_ptr5[static_cast<long>(x2)];
                        auto tmp7 = in_ptr8[static_cast<long>(x2)];
                        auto tmp12 = out_ptr4[static_cast<long>(x2)];
                        auto tmp15 = in_ptr9[static_cast<long>(x2)];
                        auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                        auto tmp5 = static_cast<float>(0.002551020408163265);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = decltype(tmp7)(tmp7 * tmp7);
                        auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                        auto tmp10 = decltype(tmp3)(tmp3 * tmp9);
                        auto tmp11 = decltype(tmp0)(tmp0 - tmp10);
                        auto tmp13 = decltype(tmp12)(tmp12 * tmp5);
                        auto tmp14 = decltype(tmp11)(tmp11 - tmp13);
                        auto tmp16 = decltype(tmp7)(tmp7 * tmp15);
                        auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                        out_ptr7[static_cast<long>(x1 + (49L*x2) + (37632L*x0))] = tmp17;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_27 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 - tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp4 = static_cast<float>(0.0006377551020408163);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp7 * tmp7;
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    auto tmp11 = tmp1 - tmp10;
                    auto tmp13 = tmp12 * tmp5;
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp16 = tmp7 * tmp15;
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bmm_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*(static_cast<long>(x1) % static_cast<long>(6L))) + (384L*x0) + (75264L*(c10::div_floor_integer(x1, 6L)))));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (3072L*x0)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (3072L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (1176L*x2) + (230496L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x3) + (1176L*x2) + (230496L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (196L*x1) + (1176L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (1176L*x2) + (230496L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (1176L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.125);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x3) + (1176L*x2) + (230496L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (1176L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0))] = tmp7;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_30 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x0 + (8L*(c10::div_floor_integer(x1, 384L))));
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(8);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((64L*x2) + (64L*x2_inner) + (12544L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(6L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))) + (static_cast<long>(x1) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(16);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-602112L) + x2 + (196L*(static_cast<long>(x1) % static_cast<long>(384L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L)))), to_float_mask(tmp11));
                            return tmp13;
                        }
                        ;
                        auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<int>(24);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((-1204224L) + (64L*x2) + (64L*x2_inner) + (12544L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(6L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))) + (static_cast<long>(x1) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp15)); })();
                            return tmp19;
                        }
                        ;
                        auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                        auto tmp21 = to_float_mask(tmp11);
                        auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                        auto tmp23 = to_float_mask(tmp4);
                        auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                        tmp24.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (225792L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x0 + (8L*(c10::div_floor_integer(x1, 384L))));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(8);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((64L*x2) + (12544L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(6L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(16);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-602112L) + x2 + (196L*(static_cast<long>(x1) % static_cast<long>(384L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(24);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr2[static_cast<long>((-1204224L) + (64L*x2) + (12544L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(6L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp11 ? tmp14 : tmp20;
                        auto tmp22 = tmp4 ? tmp7 : tmp21;
                        out_ptr0[static_cast<long>(x2 + (196L*x1) + (225792L*x0))] = tmp22;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_31 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp0 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(0.0006377551020408163);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp0 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_33 = async_compile.cpp('''
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
                       float* out_ptr2)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp0 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 - tmp5;
                    auto tmp8 = static_cast<float>(0.0006377551020408163);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp12 = tmp11 * tmp11;
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp6 * tmp13;
                    auto tmp15 = tmp1 - tmp14;
                    auto tmp17 = tmp16 * tmp9;
                    auto tmp18 = tmp15 - tmp17;
                    auto tmp20 = tmp11 * tmp19;
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bmm_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*(static_cast<long>(x1) % static_cast<long>(6L))) + (384L*x0) + (75264L*(c10::div_floor_integer(x1, 6L)))));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (3072L*x0)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (3072L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (1176L*x2) + (230496L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x3) + (1176L*x2) + (230496L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (196L*x1) + (1176L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (1176L*x2) + (230496L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (1176L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.125);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x3) + (1176L*x2) + (230496L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (1176L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0))] = tmp7;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_36 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x0 + (8L*(c10::div_floor_integer(x1, 384L))));
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(8);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((64L*x2) + (64L*x2_inner) + (12544L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(6L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))) + (static_cast<long>(x1) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(16);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-602112L) + x2 + (196L*(static_cast<long>(x1) % static_cast<long>(384L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L)))), to_float_mask(tmp11));
                            return tmp13;
                        }
                        ;
                        auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<int>(24);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((-1204224L) + (64L*x2) + (64L*x2_inner) + (12544L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(6L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))) + (static_cast<long>(x1) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp15)); })();
                            return tmp19;
                        }
                        ;
                        auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                        auto tmp21 = to_float_mask(tmp11);
                        auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                        auto tmp23 = to_float_mask(tmp4);
                        auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                        tmp24.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (225792L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x0 + (8L*(c10::div_floor_integer(x1, 384L))));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(8);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((64L*x2) + (12544L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(6L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(16);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-602112L) + x2 + (196L*(static_cast<long>(x1) % static_cast<long>(384L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(24);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr2[static_cast<long>((-1204224L) + (64L*x2) + (12544L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(6L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp11 ? tmp14 : tmp20;
                        auto tmp22 = tmp4 ? tmp7 : tmp21;
                        out_ptr0[static_cast<long>(x2 + (196L*x1) + (225792L*x0))] = tmp22;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_37 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = tmp2 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006377551020408163);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = tmp9 * tmp9;
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp1 - tmp12;
                    auto tmp15 = tmp14 * tmp7;
                    auto tmp16 = tmp13 - tmp15;
                    auto tmp18 = tmp9 * tmp17;
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp0 + tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_39 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 - tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp4 = static_cast<float>(0.0006377551020408163);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp7 * tmp7;
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    auto tmp11 = tmp1 - tmp10;
                    auto tmp13 = tmp12 * tmp5;
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp16 = tmp7 * tmp15;
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bmm_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*(static_cast<long>(x1) % static_cast<long>(6L))) + (384L*x0) + (75264L*(c10::div_floor_integer(x1, 6L)))));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (3072L*x0)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (3072L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (1176L*x2) + (230496L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x3) + (1176L*x2) + (230496L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (196L*x1) + (1176L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (1176L*x2) + (230496L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (1176L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.125);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x3) + (1176L*x2) + (230496L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (1176L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0))] = tmp7;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_42 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x0 + (8L*(c10::div_floor_integer(x1, 384L))));
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(8);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((64L*x2) + (64L*x2_inner) + (12544L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(6L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))) + (static_cast<long>(x1) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(16);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-602112L) + x2 + (196L*(static_cast<long>(x1) % static_cast<long>(384L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L)))), to_float_mask(tmp11));
                            return tmp13;
                        }
                        ;
                        auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<int>(24);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((-1204224L) + (64L*x2) + (64L*x2_inner) + (12544L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(6L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))) + (static_cast<long>(x1) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp15)); })();
                            return tmp19;
                        }
                        ;
                        auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                        auto tmp21 = to_float_mask(tmp11);
                        auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                        auto tmp23 = to_float_mask(tmp4);
                        auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                        tmp24.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (225792L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x0 + (8L*(c10::div_floor_integer(x1, 384L))));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(8);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((64L*x2) + (12544L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(6L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(16);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-602112L) + x2 + (196L*(static_cast<long>(x1) % static_cast<long>(384L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(24);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr2[static_cast<long>((-1204224L) + (64L*x2) + (12544L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(6L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp11 ? tmp14 : tmp20;
                        auto tmp22 = tmp4 ? tmp7 : tmp21;
                        out_ptr0[static_cast<long>(x2 + (196L*x1) + (225792L*x0))] = tmp22;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_43 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp0 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(0.0006377551020408163);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp0 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_45 = async_compile.cpp('''
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
                       float* out_ptr2)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp0 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 - tmp5;
                    auto tmp8 = static_cast<float>(0.0006377551020408163);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp12 = tmp11 * tmp11;
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp6 * tmp13;
                    auto tmp15 = tmp1 - tmp14;
                    auto tmp17 = tmp16 * tmp9;
                    auto tmp18 = tmp15 - tmp17;
                    auto tmp20 = tmp11 * tmp19;
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bmm_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*(static_cast<long>(x1) % static_cast<long>(6L))) + (384L*x0) + (75264L*(c10::div_floor_integer(x1, 6L)))));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (3072L*x0)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (3072L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (1176L*x2) + (230496L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x3) + (1176L*x2) + (230496L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (196L*x1) + (1176L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (6L*x3) + (6L*x3_inner) + (1176L*x2) + (230496L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (1176L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.125);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (6L*x3) + (1176L*x2) + (230496L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (1176L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            auto tmp6 = static_cast<float>(0.125);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (230496L*x0))] = tmp7;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_48 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x0 + (8L*(c10::div_floor_integer(x1, 384L))));
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(8);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((64L*x2) + (64L*x2_inner) + (12544L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(6L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))) + (static_cast<long>(x1) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(16);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-602112L) + x2 + (196L*(static_cast<long>(x1) % static_cast<long>(384L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L)))), to_float_mask(tmp11));
                            return tmp13;
                        }
                        ;
                        auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<int>(24);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((-1204224L) + (64L*x2) + (64L*x2_inner) + (12544L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(6L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))) + (static_cast<long>(x1) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp15)); })();
                            return tmp19;
                        }
                        ;
                        auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                        auto tmp21 = to_float_mask(tmp11);
                        auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                        auto tmp23 = to_float_mask(tmp4);
                        auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                        tmp24.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (225792L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x0 + (8L*(c10::div_floor_integer(x1, 384L))));
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(8);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((64L*x2) + (12544L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(6L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(16);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = in_ptr1[static_cast<long>((-602112L) + x2 + (196L*(static_cast<long>(x1) % static_cast<long>(384L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))))];
                            return tmp13;
                        }
                        ;
                        auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<long>(24);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr2[static_cast<long>((-1204224L) + (64L*x2) + (12544L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(6L))) + (75264L*x0) + (602112L*(c10::div_floor_integer(x1, 384L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp11 ? tmp14 : tmp20;
                        auto tmp22 = tmp4 ? tmp7 : tmp21;
                        out_ptr0[static_cast<long>(x2 + (196L*x1) + (225792L*x0))] = tmp22;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_sum_49 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (384L*x1)));
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp4 = tmp2 - tmp3;
                    auto tmp6 = static_cast<float>(0.0006377551020408163);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = tmp9 * tmp9;
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp1 - tmp12;
                    auto tmp15 = tmp14 * tmp7;
                    auto tmp16 = tmp13 - tmp15;
                    auto tmp18 = tmp9 * tmp17;
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp0 + tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>(x0 + (384L*x1) + (384L*x1_inner) + (75264L*x2))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x0 + (384L*x1) + (75264L*x2))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr3[static_cast<long>(x1 + (196L*x0))] = tmp_acc0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (384L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_52 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp0 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(0.00015943877551020407);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp0 - tmp15;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_55 = async_compile.cpp('''
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
                       float* out_ptr2)
{
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp0 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 - tmp5;
                    auto tmp8 = static_cast<float>(0.00015943877551020407);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp12 = tmp11 * tmp11;
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp6 * tmp13;
                    auto tmp15 = tmp1 - tmp14;
                    auto tmp17 = tmp16 * tmp9;
                    auto tmp18 = tmp15 - tmp17;
                    auto tmp20 = tmp11 * tmp19;
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_58 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = tmp2 - tmp3;
                    auto tmp6 = static_cast<float>(0.00015943877551020407);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = tmp9 * tmp9;
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp1 - tmp12;
                    auto tmp15 = tmp14 * tmp7;
                    auto tmp16 = tmp13 - tmp15;
                    auto tmp18 = tmp9 * tmp17;
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp0 + tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
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
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 - tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp4 = static_cast<float>(0.00015943877551020407);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp7 * tmp7;
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    auto tmp11 = tmp1 - tmp10;
                    auto tmp13 = tmp12 * tmp5;
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp16 = tmp7 * tmp15;
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_64 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp0 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(0.00015943877551020407);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp0 - tmp15;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_67 = async_compile.cpp('''
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
                       float* out_ptr2)
{
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp0 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 - tmp5;
                    auto tmp8 = static_cast<float>(0.00015943877551020407);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp12 = tmp11 * tmp11;
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp6 * tmp13;
                    auto tmp15 = tmp1 - tmp14;
                    auto tmp17 = tmp16 * tmp9;
                    auto tmp18 = tmp15 - tmp17;
                    auto tmp20 = tmp11 * tmp19;
                    auto tmp21 = tmp18 * tmp20;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_gelu_gelu_backward_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.7071067811865476);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp4.erf();
                auto tmp6 = static_cast<float>(1.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = static_cast<float>(0.5);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp8 * tmp10;
                auto tmp12 = tmp1 * tmp1;
                auto tmp13 = static_cast<float>(-0.5);
                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                auto tmp15 = tmp12 * tmp14;
                auto tmp16 = tmp15.exp();
                auto tmp17 = static_cast<float>(0.3989422804014327);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 * tmp18;
                auto tmp20 = tmp1 * tmp19;
                auto tmp21 = tmp11 + tmp20;
                auto tmp22 = tmp0 * tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_sum_70 = async_compile.cpp('''
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp4 = tmp2 - tmp3;
                    auto tmp6 = static_cast<float>(0.00015943877551020407);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = tmp9 * tmp9;
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp13 = tmp1 - tmp12;
                    auto tmp15 = tmp14 * tmp7;
                    auto tmp16 = tmp13 - tmp15;
                    auto tmp18 = tmp9 * tmp17;
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp0 + tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>(x0 + (192L*x1) + (192L*x1_inner) + (150528L*x2))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (784L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_71 = async_compile.cpp('''
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_4, primals_5, primals_7, primals_9, primals_11, primals_13, primals_14, primals_15, primals_16, primals_18, primals_19, primals_20, primals_21, primals_23, primals_24, primals_25, primals_26, primals_28, primals_29, primals_30, primals_31, primals_33, primals_34, primals_35, primals_36, primals_38, primals_39, primals_40, primals_41, primals_43, primals_44, primals_45, primals_46, primals_48, primals_50, primals_52, primals_53, primals_54, primals_56, primals_57, primals_58, primals_60, primals_61, primals_62, primals_64, primals_65, primals_66, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_76, primals_77, primals_78, primals_80, primals_81, primals_82, primals_84, primals_86, primals_88, primals_89, primals_90, primals_92, primals_93, primals_94, primals_96, primals_97, primals_98, primals_100, primals_101, primals_102, primals_104, primals_105, primals_106, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_116, primals_117, primals_118, primals_206, convolution, squeeze_1, relu, convolution_1, squeeze_4, clone, squeeze_7, add_15, convolution_2, clone_1, convolution_3, mul_26, convolution_4, squeeze_10, add_23, convolution_5, clone_3, convolution_6, mul_39, convolution_7, squeeze_13, add_31, convolution_8, clone_5, convolution_9, mul_52, convolution_10, squeeze_16, add_39, convolution_11, clone_7, convolution_12, mul_65, convolution_13, squeeze_19, add_47, convolution_14, clone_9, convolution_15, mul_78, convolution_16, squeeze_22, add_55, convolution_17, clone_11, convolution_18, mul_91, convolution_19, squeeze_25, add_63, convolution_20, clone_13, convolution_21, mul_104, add_66, convolution_23, squeeze_28, clone_15, squeeze_31, add_77, view_7, convolution_25, squeeze_34, add_83, convolution_26, clone_22, convolution_27, squeeze_37, add_90, view_15, convolution_29, squeeze_40, add_96, convolution_30, clone_30, convolution_31, squeeze_43, add_103, view_23, convolution_33, squeeze_46, add_109, convolution_34, clone_38, convolution_35, squeeze_49, add_116, view_31, convolution_37, squeeze_52, add_122, convolution_38, clone_46, add_124, convolution_40, squeeze_55, clone_48, squeeze_58, add_135, view_39, convolution_42, squeeze_61, add_141, convolution_43, clone_55, convolution_44, squeeze_64, add_148, view_47, convolution_46, squeeze_67, add_154, convolution_47, clone_63, convolution_48, squeeze_70, add_161, view_55, convolution_50, squeeze_73, add_167, convolution_51, clone_71, convolution_52, squeeze_76, add_174, view_63, convolution_54, squeeze_79, add_180, convolution_55, clone_79, convolution_56, squeeze_82, clone_81, permute_25, unsqueeze_114, unsqueeze_126, permute_30, permute_31, alias_9, permute_32, permute_33, unsqueeze_138, unsqueeze_150, permute_37, permute_38, alias_10, permute_39, permute_40, unsqueeze_162, unsqueeze_174, permute_44, permute_45, alias_11, permute_46, permute_47, unsqueeze_186, unsqueeze_198, permute_51, permute_52, alias_12, permute_53, permute_54, unsqueeze_210, unsqueeze_222, unsqueeze_234, permute_58, permute_59, alias_13, permute_60, permute_61, unsqueeze_246, unsqueeze_258, permute_65, permute_66, alias_14, permute_67, permute_68, unsqueeze_270, unsqueeze_282, permute_72, permute_73, alias_15, permute_74, permute_75, unsqueeze_294, unsqueeze_306, permute_79, permute_80, alias_16, permute_81, permute_82, unsqueeze_318, unsqueeze_330, unsqueeze_342, unsqueeze_354, unsqueeze_366, unsqueeze_378, unsqueeze_390, unsqueeze_402, unsqueeze_414, unsqueeze_426, unsqueeze_438, tangents_1 = args
    args.clear()
    assert_size_stride(primals_4, (32, 3, 7, 7), (147, 1, 21, 3))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_7, (192, 32, 4, 4), (512, 1, 128, 32))
    assert_size_stride(primals_9, (192, ), (1, ))
    assert_size_stride(primals_11, (192, ), (1, ))
    assert_size_stride(primals_13, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_14, (384, 48, 3, 3), (432, 1, 144, 48))
    assert_size_stride(primals_15, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_16, (192, ), (1, ))
    assert_size_stride(primals_18, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_19, (384, 48, 3, 3), (432, 1, 144, 48))
    assert_size_stride(primals_20, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_21, (192, ), (1, ))
    assert_size_stride(primals_23, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_24, (384, 48, 3, 3), (432, 1, 144, 48))
    assert_size_stride(primals_25, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_26, (192, ), (1, ))
    assert_size_stride(primals_28, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_29, (384, 48, 3, 3), (432, 1, 144, 48))
    assert_size_stride(primals_30, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_33, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_34, (384, 48, 3, 3), (432, 1, 144, 48))
    assert_size_stride(primals_35, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_36, (192, ), (1, ))
    assert_size_stride(primals_38, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_39, (384, 48, 3, 3), (432, 1, 144, 48))
    assert_size_stride(primals_40, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_41, (192, ), (1, ))
    assert_size_stride(primals_43, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_44, (384, 48, 3, 3), (432, 1, 144, 48))
    assert_size_stride(primals_45, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_46, (384, 192, 2, 2), (768, 1, 384, 192))
    assert_size_stride(primals_48, (384, ), (1, ))
    assert_size_stride(primals_50, (384, ), (1, ))
    assert_size_stride(primals_52, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_53, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_54, (384, ), (1, ))
    assert_size_stride(primals_56, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_57, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_58, (384, ), (1, ))
    assert_size_stride(primals_60, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_61, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_62, (384, ), (1, ))
    assert_size_stride(primals_64, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_65, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_66, (384, ), (1, ))
    assert_size_stride(primals_68, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_69, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_70, (384, ), (1, ))
    assert_size_stride(primals_72, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_73, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_74, (384, ), (1, ))
    assert_size_stride(primals_76, (1152, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_77, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_80, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_81, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_82, (768, 384, 2, 2), (1536, 1, 768, 384))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_88, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_89, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_92, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_93, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_96, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_97, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_100, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_101, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_104, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_105, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_108, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_109, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_112, (2304, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_113, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_116, (3072, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_117, (768, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_206, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(relu, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_1, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_4, (192, ), (1, ))
    assert_size_stride(clone, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_7, (192, ), (1, ))
    assert_size_stride(add_15, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_2, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(clone_1, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(convolution_3, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(mul_26, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(convolution_4, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_10, (192, ), (1, ))
    assert_size_stride(add_23, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_5, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(clone_3, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(convolution_6, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(mul_39, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(convolution_7, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_13, (192, ), (1, ))
    assert_size_stride(add_31, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_8, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(clone_5, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(convolution_9, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(mul_52, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(convolution_10, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_16, (192, ), (1, ))
    assert_size_stride(add_39, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_11, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(clone_7, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(convolution_12, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(mul_65, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(convolution_13, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_19, (192, ), (1, ))
    assert_size_stride(add_47, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_14, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(clone_9, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(convolution_15, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(mul_78, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(convolution_16, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_22, (192, ), (1, ))
    assert_size_stride(add_55, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_17, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(clone_11, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(convolution_18, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(mul_91, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(convolution_19, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_25, (192, ), (1, ))
    assert_size_stride(add_63, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_20, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(clone_13, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(convolution_21, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(mul_104, (8, 384, 28, 28), (301056, 1, 10752, 384))
    assert_size_stride(add_66, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_23, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_28, (384, ), (1, ))
    assert_size_stride(clone_15, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_31, (384, ), (1, ))
    assert_size_stride(add_77, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(view_7, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_25, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_34, (384, ), (1, ))
    assert_size_stride(add_83, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_26, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_22, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_27, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_37, (384, ), (1, ))
    assert_size_stride(add_90, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(view_15, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_29, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_40, (384, ), (1, ))
    assert_size_stride(add_96, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_30, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_30, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_31, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_43, (384, ), (1, ))
    assert_size_stride(add_103, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(view_23, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_33, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_46, (384, ), (1, ))
    assert_size_stride(add_109, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_34, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_38, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(convolution_35, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_49, (384, ), (1, ))
    assert_size_stride(add_116, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(view_31, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_37, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_52, (384, ), (1, ))
    assert_size_stride(add_122, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_38, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(clone_46, (8, 1536, 14, 14), (301056, 1, 21504, 1536))
    assert_size_stride(add_124, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_40, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(squeeze_55, (768, ), (1, ))
    assert_size_stride(clone_48, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(squeeze_58, (768, ), (1, ))
    assert_size_stride(add_135, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(view_39, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(convolution_42, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(squeeze_61, (768, ), (1, ))
    assert_size_stride(add_141, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(convolution_43, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(clone_55, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(convolution_44, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(squeeze_64, (768, ), (1, ))
    assert_size_stride(add_148, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(view_47, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(convolution_46, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(squeeze_67, (768, ), (1, ))
    assert_size_stride(add_154, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(convolution_47, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(clone_63, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(convolution_48, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(squeeze_70, (768, ), (1, ))
    assert_size_stride(add_161, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(view_55, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(convolution_50, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(squeeze_73, (768, ), (1, ))
    assert_size_stride(add_167, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(convolution_51, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(clone_71, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(convolution_52, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(squeeze_76, (768, ), (1, ))
    assert_size_stride(add_174, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(view_63, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(convolution_54, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(squeeze_79, (768, ), (1, ))
    assert_size_stride(add_180, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(convolution_55, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(clone_79, (8, 3072, 7, 7), (150528, 1, 21504, 3072))
    assert_size_stride(convolution_56, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(squeeze_82, (768, ), (1, ))
    assert_size_stride(clone_81, (8, 768), (768, 1))
    assert_size_stride(permute_25, (1000, 768), (768, 1))
    assert_size_stride(unsqueeze_114, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_126, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(permute_30, (48, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_31, (48, 128, 49), (6272, 1, 128))
    assert_size_stride(alias_9, (8, 6, 49, 49), (14406, 1, 294, 6))
    assert_size_stride(permute_32, (48, 128, 49), (6272, 1, 128))
    assert_size_stride(permute_33, (48, 49, 128), (6272, 1, 49))
    assert_size_stride(unsqueeze_138, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_150, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(permute_37, (48, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_38, (48, 128, 49), (6272, 1, 128))
    assert_size_stride(alias_10, (8, 6, 49, 49), (14406, 1, 294, 6))
    assert_size_stride(permute_39, (48, 128, 49), (6272, 1, 128))
    assert_size_stride(permute_40, (48, 49, 128), (6272, 1, 49))
    assert_size_stride(unsqueeze_162, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_174, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(permute_44, (48, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_45, (48, 128, 49), (6272, 1, 128))
    assert_size_stride(alias_11, (8, 6, 49, 49), (14406, 1, 294, 6))
    assert_size_stride(permute_46, (48, 128, 49), (6272, 1, 128))
    assert_size_stride(permute_47, (48, 49, 128), (6272, 1, 49))
    assert_size_stride(unsqueeze_186, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_198, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(permute_51, (48, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_52, (48, 128, 49), (6272, 1, 128))
    assert_size_stride(alias_12, (8, 6, 49, 49), (14406, 1, 294, 6))
    assert_size_stride(permute_53, (48, 128, 49), (6272, 1, 128))
    assert_size_stride(permute_54, (48, 49, 128), (6272, 1, 49))
    assert_size_stride(unsqueeze_210, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_222, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_234, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(permute_58, (48, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_59, (48, 64, 196), (12544, 1, 64))
    assert_size_stride(alias_13, (8, 6, 196, 196), (230496, 1, 1176, 6))
    assert_size_stride(permute_60, (48, 64, 196), (12544, 1, 64))
    assert_size_stride(permute_61, (48, 196, 64), (12544, 1, 196))
    assert_size_stride(unsqueeze_246, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(permute_65, (48, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_66, (48, 64, 196), (12544, 1, 64))
    assert_size_stride(alias_14, (8, 6, 196, 196), (230496, 1, 1176, 6))
    assert_size_stride(permute_67, (48, 64, 196), (12544, 1, 64))
    assert_size_stride(permute_68, (48, 196, 64), (12544, 1, 196))
    assert_size_stride(unsqueeze_270, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_282, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(permute_72, (48, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_73, (48, 64, 196), (12544, 1, 64))
    assert_size_stride(alias_15, (8, 6, 196, 196), (230496, 1, 1176, 6))
    assert_size_stride(permute_74, (48, 64, 196), (12544, 1, 64))
    assert_size_stride(permute_75, (48, 196, 64), (12544, 1, 196))
    assert_size_stride(unsqueeze_294, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_306, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(permute_79, (48, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_80, (48, 64, 196), (12544, 1, 64))
    assert_size_stride(alias_16, (8, 6, 196, 196), (230496, 1, 1176, 6))
    assert_size_stride(permute_81, (48, 64, 196), (12544, 1, 64))
    assert_size_stride(permute_82, (48, 196, 64), (12544, 1, 196))
    assert_size_stride(unsqueeze_318, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_330, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_342, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_354, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_366, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_378, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_390, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_414, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_426, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_438, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_0(c_void_p(clone.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(clone_15.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(clone_48.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()))
    del convolution_13
    del convolution_31
    del convolution_48
    buf3 = empty((8, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_25, out=buf3)
    del permute_25
    buf4 = empty((1000, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_81, out=buf4)
    del clone_81
    buf5 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    buf19 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    buf6 = empty((768, ), device='cpu', dtype=torch.float32)
    buf8 = empty((768, ), device='cpu', dtype=torch.float32)
    buf9 = empty((8, 768, 7, 7), device='cpu', dtype=torch.float32)
    buf10 = buf8; del buf8  # reuse
    cpp_fused_add_div_native_batch_norm_backward_sum_1(c_void_p(buf10.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(convolution_56.data_ptr()), c_void_p(unsqueeze_114.data_ptr()), c_void_p(unsqueeze_126.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf9.data_ptr()))
    del buf3
    del buf7
    del convolution_54
    del convolution_56
    del primals_118
    del squeeze_82
    del tangents_1
    del unsqueeze_114
    del unsqueeze_126
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf11 = aten.convolution_backward(buf9, clone_79, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del clone_79
    del primals_117
    buf12 = buf11[0]
    buf13 = buf11[1]
    del buf11
    buf14 = buf12; del buf12  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_2(c_void_p(buf14.data_ptr()), c_void_p(convolution_55.data_ptr()))
    del convolution_55
    # Source Nodes: [x_162], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf15 = aten.convolution_backward(buf14, add_180, primals_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_180
    del buf14
    del primals_116
    buf16 = buf15[0]
    buf17 = buf15[1]
    del buf15
    buf18 = empty((768, ), device='cpu', dtype=torch.float32)
    buf20 = empty((768, ), device='cpu', dtype=torch.float32)
    buf21 = empty((768, ), device='cpu', dtype=torch.float32)
    buf22 = buf9; del buf9  # reuse
    cpp_fused_add_native_batch_norm_backward_3(c_void_p(buf22.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()))
    del primals_114
    del squeeze_79
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf23 = aten.convolution_backward(buf22, view_63, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_113
    del view_63
    buf24 = buf23[0]
    buf25 = buf23[1]
    del buf23
    buf26 = reinterpret_tensor(buf19, (48, 49, 128), (128, 6144, 1), 0); del buf19  # reuse
    buf28 = reinterpret_tensor(buf16, (48, 49, 128), (128, 6144, 1), 0); del buf16  # reuse
    cpp_fused_bmm_4(c_void_p(buf24.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf28.data_ptr()))
    buf27 = reinterpret_tensor(buf24, (48, 49, 128), (6272, 128, 1), 0); del buf24  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_30, buf26, out=buf27)
    del permute_30
    buf29 = empty((48, 49, 49), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf28, permute_31, out=buf29)
    del permute_31
    buf30 = empty_strided((8, 6, 49, 1), (294, 49, 1, 2352), device='cpu', dtype=torch.float32)
    buf31 = reinterpret_tensor(buf29, (8, 6, 49, 49), (14406, 2401, 49, 1), 0); del buf29  # reuse
    cpp_fused__softmax_backward_data_mul_5(c_void_p(buf31.data_ptr()), c_void_p(alias_9.data_ptr()), c_void_p(buf30.data_ptr()))
    del alias_9
    buf32 = reinterpret_tensor(buf28, (48, 128, 49), (6272, 49, 1), 0); del buf28  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_32, reinterpret_tensor(buf31, (48, 49, 49), (2401, 49, 1), 0), out=buf32)
    del permute_32
    buf33 = reinterpret_tensor(buf26, (48, 49, 128), (6272, 128, 1), 0); del buf26  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf31, (48, 49, 49), (2401, 49, 1), 0), permute_33, out=buf33)
    del permute_33
    buf34 = empty((8, 2304, 7, 7), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_6(c_void_p(buf33.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf34.data_ptr()))
    del buf27
    del buf32
    del buf33
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf35 = aten.convolution_backward(buf34, add_174, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_174
    del primals_112
    buf36 = buf35[0]
    buf37 = buf35[1]
    del buf35
    buf38 = buf20; del buf20  # reuse
    buf39 = empty((768, ), device='cpu', dtype=torch.float32)
    buf40 = buf36; del buf36  # reuse
    buf41 = buf39; del buf39  # reuse
    buf42 = buf22; del buf22  # reuse
    cpp_fused_add_native_batch_norm_backward_7(c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(unsqueeze_138.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf38.data_ptr()))
    del convolution_52
    del primals_110
    del squeeze_76
    del unsqueeze_138
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf43 = aten.convolution_backward(buf42, clone_71, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del clone_71
    del primals_109
    buf44 = buf43[0]
    buf45 = buf43[1]
    del buf43
    buf46 = buf44; del buf44  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_8(c_void_p(buf46.data_ptr()), c_void_p(convolution_51.data_ptr()))
    del convolution_51
    # Source Nodes: [x_150], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf47 = aten.convolution_backward(buf46, add_167, primals_108, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_167
    del buf46
    del primals_108
    buf48 = buf47[0]
    buf49 = buf47[1]
    del buf47
    buf50 = empty((768, ), device='cpu', dtype=torch.float32)
    buf51 = empty((768, ), device='cpu', dtype=torch.float32)
    buf52 = empty((768, ), device='cpu', dtype=torch.float32)
    buf53 = buf42; del buf42  # reuse
    cpp_fused_add_native_batch_norm_backward_9(c_void_p(buf53.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(unsqueeze_150.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()))
    del convolution_50
    del primals_106
    del squeeze_73
    del unsqueeze_150
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf54 = aten.convolution_backward(buf53, view_55, primals_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_105
    del view_55
    buf55 = buf54[0]
    buf56 = buf54[1]
    del buf54
    buf57 = reinterpret_tensor(buf48, (48, 49, 128), (128, 6144, 1), 0); del buf48  # reuse
    buf59 = reinterpret_tensor(buf40, (48, 49, 128), (128, 6144, 1), 0); del buf40  # reuse
    cpp_fused_bmm_10(c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf59.data_ptr()))
    buf58 = reinterpret_tensor(buf55, (48, 49, 128), (6272, 128, 1), 0); del buf55  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_37, buf57, out=buf58)
    del permute_37
    buf60 = reinterpret_tensor(buf31, (48, 49, 49), (2401, 49, 1), 0); del buf31  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf59, permute_38, out=buf60)
    del permute_38
    buf61 = buf30; del buf30  # reuse
    buf62 = reinterpret_tensor(buf60, (8, 6, 49, 49), (14406, 2401, 49, 1), 0); del buf60  # reuse
    cpp_fused__softmax_backward_data_mul_11(c_void_p(buf62.data_ptr()), c_void_p(alias_10.data_ptr()), c_void_p(buf61.data_ptr()))
    del alias_10
    buf63 = reinterpret_tensor(buf59, (48, 128, 49), (6272, 49, 1), 0); del buf59  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_39, reinterpret_tensor(buf62, (48, 49, 49), (2401, 49, 1), 0), out=buf63)
    del permute_39
    buf64 = reinterpret_tensor(buf57, (48, 49, 128), (6272, 128, 1), 0); del buf57  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf62, (48, 49, 49), (2401, 49, 1), 0), permute_40, out=buf64)
    del permute_40
    buf65 = buf34; del buf34  # reuse
    cpp_fused_convolution_backward_12(c_void_p(buf64.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf65.data_ptr()))
    del buf58
    del buf63
    del buf64
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf66 = aten.convolution_backward(buf65, add_161, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_161
    del primals_104
    buf67 = buf66[0]
    buf68 = buf66[1]
    del buf66
    buf69 = buf51; del buf51  # reuse
    buf70 = empty((768, ), device='cpu', dtype=torch.float32)
    buf71 = empty((768, ), device='cpu', dtype=torch.float32)
    buf72 = buf53; del buf53  # reuse
    cpp_fused_add_native_batch_norm_backward_13(c_void_p(buf72.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(unsqueeze_162.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()))
    del buf2
    del primals_102
    del squeeze_70
    del unsqueeze_162
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf73 = aten.convolution_backward(buf72, clone_63, primals_101, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del clone_63
    del primals_101
    buf74 = buf73[0]
    buf75 = buf73[1]
    del buf73
    buf76 = buf74; del buf74  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_14(c_void_p(buf76.data_ptr()), c_void_p(convolution_47.data_ptr()))
    del convolution_47
    # Source Nodes: [x_138], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf77 = aten.convolution_backward(buf76, add_154, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_154
    del buf76
    del primals_100
    buf78 = buf77[0]
    buf79 = buf77[1]
    del buf77
    buf81 = buf67; del buf67  # reuse
    buf80 = buf70; del buf70  # reuse
    buf82 = empty((768, ), device='cpu', dtype=torch.float32)
    buf83 = empty((768, ), device='cpu', dtype=torch.float32)
    buf84 = buf72; del buf72  # reuse
    cpp_fused_add_native_batch_norm_backward_15(c_void_p(buf84.data_ptr()), c_void_p(clone_48.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(unsqueeze_174.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()))
    del convolution_46
    del primals_98
    del squeeze_67
    del unsqueeze_174
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf85 = aten.convolution_backward(buf84, view_47, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_97
    del view_47
    buf86 = buf85[0]
    buf87 = buf85[1]
    del buf85
    buf88 = reinterpret_tensor(buf81, (48, 49, 128), (128, 6144, 1), 0); del buf81  # reuse
    buf90 = reinterpret_tensor(buf78, (48, 49, 128), (128, 6144, 1), 0); del buf78  # reuse
    cpp_fused_bmm_16(c_void_p(buf86.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf90.data_ptr()))
    buf89 = reinterpret_tensor(buf86, (48, 49, 128), (6272, 128, 1), 0); del buf86  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_44, buf88, out=buf89)
    del permute_44
    buf91 = reinterpret_tensor(buf62, (48, 49, 49), (2401, 49, 1), 0); del buf62  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf90, permute_45, out=buf91)
    del permute_45
    buf92 = buf61; del buf61  # reuse
    buf93 = reinterpret_tensor(buf91, (8, 6, 49, 49), (14406, 2401, 49, 1), 0); del buf91  # reuse
    cpp_fused__softmax_backward_data_mul_17(c_void_p(buf93.data_ptr()), c_void_p(alias_11.data_ptr()), c_void_p(buf92.data_ptr()))
    del alias_11
    buf94 = reinterpret_tensor(buf90, (48, 128, 49), (6272, 49, 1), 0); del buf90  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_46, reinterpret_tensor(buf93, (48, 49, 49), (2401, 49, 1), 0), out=buf94)
    del permute_46
    buf95 = reinterpret_tensor(buf88, (48, 49, 128), (6272, 128, 1), 0); del buf88  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf93, (48, 49, 49), (2401, 49, 1), 0), permute_47, out=buf95)
    del permute_47
    buf96 = buf65; del buf65  # reuse
    cpp_fused_convolution_backward_18(c_void_p(buf95.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf96.data_ptr()))
    del buf89
    del buf94
    del buf95
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf97 = aten.convolution_backward(buf96, add_148, primals_96, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_148
    del primals_96
    buf98 = buf97[0]
    buf99 = buf97[1]
    del buf97
    buf100 = buf82; del buf82  # reuse
    buf101 = empty((768, ), device='cpu', dtype=torch.float32)
    buf102 = buf98; del buf98  # reuse
    buf103 = buf101; del buf101  # reuse
    buf104 = buf84; del buf84  # reuse
    cpp_fused_add_native_batch_norm_backward_19(c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(clone_48.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(unsqueeze_186.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(buf100.data_ptr()))
    del convolution_44
    del primals_94
    del squeeze_64
    del unsqueeze_186
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf105 = aten.convolution_backward(buf104, clone_55, primals_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del clone_55
    del primals_93
    buf106 = buf105[0]
    buf107 = buf105[1]
    del buf105
    buf108 = buf106; del buf106  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_20(c_void_p(buf108.data_ptr()), c_void_p(convolution_43.data_ptr()))
    del convolution_43
    # Source Nodes: [x_126], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf109 = aten.convolution_backward(buf108, add_141, primals_92, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_141
    del buf108
    del primals_92
    buf110 = buf109[0]
    buf111 = buf109[1]
    del buf109
    buf112 = empty((768, ), device='cpu', dtype=torch.float32)
    buf113 = empty((768, ), device='cpu', dtype=torch.float32)
    buf114 = empty((768, ), device='cpu', dtype=torch.float32)
    buf115 = buf104; del buf104  # reuse
    cpp_fused_add_native_batch_norm_backward_21(c_void_p(buf115.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(clone_48.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(unsqueeze_198.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()))
    del convolution_42
    del primals_90
    del squeeze_61
    del unsqueeze_198
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf116 = aten.convolution_backward(buf115, view_39, primals_89, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_89
    del view_39
    buf117 = buf116[0]
    buf118 = buf116[1]
    del buf116
    buf119 = reinterpret_tensor(buf110, (48, 49, 128), (128, 6144, 1), 0); del buf110  # reuse
    buf121 = reinterpret_tensor(buf102, (48, 49, 128), (128, 6144, 1), 0); del buf102  # reuse
    cpp_fused_bmm_22(c_void_p(buf117.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf121.data_ptr()))
    buf120 = reinterpret_tensor(buf117, (48, 49, 128), (6272, 128, 1), 0); del buf117  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_51, buf119, out=buf120)
    del permute_51
    buf122 = reinterpret_tensor(buf93, (48, 49, 49), (2401, 49, 1), 0); del buf93  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf121, permute_52, out=buf122)
    del permute_52
    buf123 = buf92; del buf92  # reuse
    buf124 = reinterpret_tensor(buf122, (8, 6, 49, 49), (14406, 2401, 49, 1), 0); del buf122  # reuse
    cpp_fused__softmax_backward_data_mul_23(c_void_p(buf124.data_ptr()), c_void_p(alias_12.data_ptr()), c_void_p(buf123.data_ptr()))
    del alias_12
    del buf123
    buf125 = reinterpret_tensor(buf121, (48, 128, 49), (6272, 49, 1), 0); del buf121  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_53, reinterpret_tensor(buf124, (48, 49, 49), (2401, 49, 1), 0), out=buf125)
    del permute_53
    buf126 = reinterpret_tensor(buf119, (48, 49, 128), (6272, 128, 1), 0); del buf119  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf124, (48, 49, 49), (2401, 49, 1), 0), permute_54, out=buf126)
    del buf124
    del permute_54
    buf127 = buf96; del buf96  # reuse
    cpp_fused_convolution_backward_24(c_void_p(buf126.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf127.data_ptr()))
    del buf120
    del buf125
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf128 = aten.convolution_backward(buf127, add_135, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_135
    del buf127
    del primals_88
    buf129 = buf128[0]
    buf130 = buf128[1]
    del buf128
    buf131 = buf113; del buf113  # reuse
    buf132 = empty((768, ), device='cpu', dtype=torch.float32)
    buf133 = empty((768, ), device='cpu', dtype=torch.float32)
    buf134 = buf129; del buf129  # reuse
    buf135 = empty((1, 768, 7, 7), device='cpu', dtype=torch.float32)
    buf136 = empty((768, ), device='cpu', dtype=torch.float32)
    buf137 = empty((768, ), device='cpu', dtype=torch.float32)
    buf138 = empty((768, ), device='cpu', dtype=torch.float32)
    buf139 = reinterpret_tensor(buf126, (8, 768, 7, 7), (37632, 49, 7, 1), 0); del buf126  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_sum_25(c_void_p(buf134.data_ptr()), c_void_p(clone_48.data_ptr()), c_void_p(unsqueeze_210.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(unsqueeze_222.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()))
    del buf115
    del buf132
    del buf134
    del buf137
    del clone_48
    del convolution_40
    del primals_84
    del primals_86
    del squeeze_55
    del squeeze_58
    del unsqueeze_210
    del unsqueeze_222
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf140 = aten.convolution_backward(buf139, add_124, primals_82, [768], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_124
    del buf139
    del primals_82
    buf141 = buf140[0]
    buf142 = buf140[1]
    buf143 = buf140[2]
    del buf140
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf144 = aten.convolution_backward(buf141, clone_46, primals_81, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del clone_46
    del primals_81
    buf145 = buf144[0]
    buf146 = buf144[1]
    del buf144
    buf147 = buf145; del buf145  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_26(c_void_p(buf147.data_ptr()), c_void_p(convolution_38.data_ptr()))
    del convolution_38
    # Source Nodes: [x_109], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf148 = aten.convolution_backward(buf147, add_122, primals_80, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_122
    del buf147
    del primals_80
    buf149 = buf148[0]
    buf150 = buf148[1]
    del buf148
    buf152 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    buf151 = empty((384, ), device='cpu', dtype=torch.float32)
    buf153 = empty((384, ), device='cpu', dtype=torch.float32)
    buf154 = empty((384, ), device='cpu', dtype=torch.float32)
    buf155 = buf141; del buf141  # reuse
    cpp_fused_add_native_batch_norm_backward_27(c_void_p(buf155.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(unsqueeze_234.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()))
    del convolution_37
    del primals_78
    del squeeze_52
    del unsqueeze_234
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf156 = aten.convolution_backward(buf155, view_31, primals_77, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_77
    del view_31
    buf157 = buf156[0]
    buf158 = buf156[1]
    del buf156
    buf159 = reinterpret_tensor(buf152, (48, 196, 64), (64, 3072, 1), 0); del buf152  # reuse
    buf161 = reinterpret_tensor(buf149, (48, 196, 64), (64, 3072, 1), 0); del buf149  # reuse
    cpp_fused_bmm_28(c_void_p(buf157.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf161.data_ptr()))
    buf160 = reinterpret_tensor(buf157, (48, 196, 64), (12544, 64, 1), 0); del buf157  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_58, buf159, out=buf160)
    del permute_58
    buf162 = empty((48, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf161, permute_59, out=buf162)
    del permute_59
    buf163 = empty_strided((8, 6, 196, 1), (1176, 196, 1, 9408), device='cpu', dtype=torch.float32)
    buf164 = reinterpret_tensor(buf162, (8, 6, 196, 196), (230496, 38416, 196, 1), 0); del buf162  # reuse
    cpp_fused__softmax_backward_data_mul_29(c_void_p(buf164.data_ptr()), c_void_p(alias_13.data_ptr()), c_void_p(buf163.data_ptr()))
    del alias_13
    buf165 = reinterpret_tensor(buf161, (48, 64, 196), (12544, 196, 1), 0); del buf161  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_60, reinterpret_tensor(buf164, (48, 196, 196), (38416, 196, 1), 0), out=buf165)
    del permute_60
    buf166 = reinterpret_tensor(buf159, (48, 196, 64), (12544, 64, 1), 0); del buf159  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf164, (48, 196, 196), (38416, 196, 1), 0), permute_61, out=buf166)
    del permute_61
    buf167 = empty((8, 1152, 14, 14), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_30(c_void_p(buf166.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf167.data_ptr()))
    del buf160
    del buf165
    del buf166
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf168 = aten.convolution_backward(buf167, add_116, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_116
    del primals_76
    buf169 = buf168[0]
    buf170 = buf168[1]
    del buf168
    buf171 = buf153; del buf153  # reuse
    buf172 = empty((384, ), device='cpu', dtype=torch.float32)
    buf173 = buf169; del buf169  # reuse
    buf174 = buf172; del buf172  # reuse
    buf175 = buf155; del buf155  # reuse
    cpp_fused_add_native_batch_norm_backward_31(c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_246.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf171.data_ptr()))
    del convolution_35
    del primals_74
    del squeeze_49
    del unsqueeze_246
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf176 = aten.convolution_backward(buf175, clone_38, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del clone_38
    del primals_73
    buf177 = buf176[0]
    buf178 = buf176[1]
    del buf176
    buf179 = buf177; del buf177  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_32(c_void_p(buf179.data_ptr()), c_void_p(convolution_34.data_ptr()))
    del convolution_34
    # Source Nodes: [x_97], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf180 = aten.convolution_backward(buf179, add_109, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_109
    del buf179
    del primals_72
    buf181 = buf180[0]
    buf182 = buf180[1]
    del buf180
    buf183 = empty((384, ), device='cpu', dtype=torch.float32)
    buf184 = empty((384, ), device='cpu', dtype=torch.float32)
    buf185 = empty((384, ), device='cpu', dtype=torch.float32)
    buf186 = buf175; del buf175  # reuse
    cpp_fused_add_native_batch_norm_backward_33(c_void_p(buf186.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_258.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()))
    del convolution_33
    del primals_70
    del squeeze_46
    del unsqueeze_258
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf187 = aten.convolution_backward(buf186, view_23, primals_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_69
    del view_23
    buf188 = buf187[0]
    buf189 = buf187[1]
    del buf187
    buf190 = reinterpret_tensor(buf181, (48, 196, 64), (64, 3072, 1), 0); del buf181  # reuse
    buf192 = reinterpret_tensor(buf173, (48, 196, 64), (64, 3072, 1), 0); del buf173  # reuse
    cpp_fused_bmm_34(c_void_p(buf188.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf192.data_ptr()))
    buf191 = reinterpret_tensor(buf188, (48, 196, 64), (12544, 64, 1), 0); del buf188  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_65, buf190, out=buf191)
    del permute_65
    buf193 = reinterpret_tensor(buf164, (48, 196, 196), (38416, 196, 1), 0); del buf164  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf192, permute_66, out=buf193)
    del permute_66
    buf194 = buf163; del buf163  # reuse
    buf195 = reinterpret_tensor(buf193, (8, 6, 196, 196), (230496, 38416, 196, 1), 0); del buf193  # reuse
    cpp_fused__softmax_backward_data_mul_35(c_void_p(buf195.data_ptr()), c_void_p(alias_14.data_ptr()), c_void_p(buf194.data_ptr()))
    del alias_14
    buf196 = reinterpret_tensor(buf192, (48, 64, 196), (12544, 196, 1), 0); del buf192  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_67, reinterpret_tensor(buf195, (48, 196, 196), (38416, 196, 1), 0), out=buf196)
    del permute_67
    buf197 = reinterpret_tensor(buf190, (48, 196, 64), (12544, 64, 1), 0); del buf190  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf195, (48, 196, 196), (38416, 196, 1), 0), permute_68, out=buf197)
    del permute_68
    buf198 = buf167; del buf167  # reuse
    cpp_fused_convolution_backward_36(c_void_p(buf197.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf198.data_ptr()))
    del buf191
    del buf196
    del buf197
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf199 = aten.convolution_backward(buf198, add_103, primals_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_103
    del primals_68
    buf200 = buf199[0]
    buf201 = buf199[1]
    del buf199
    buf202 = buf184; del buf184  # reuse
    buf203 = empty((384, ), device='cpu', dtype=torch.float32)
    buf204 = empty((384, ), device='cpu', dtype=torch.float32)
    buf205 = buf1; del buf1  # reuse
    cpp_fused_add_native_batch_norm_backward_37(c_void_p(buf205.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(unsqueeze_270.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()))
    del buf186
    del primals_66
    del squeeze_43
    del unsqueeze_270
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf206 = aten.convolution_backward(buf205, clone_30, primals_65, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del clone_30
    del primals_65
    buf207 = buf206[0]
    buf208 = buf206[1]
    del buf206
    buf209 = buf207; del buf207  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_38(c_void_p(buf209.data_ptr()), c_void_p(convolution_30.data_ptr()))
    del convolution_30
    # Source Nodes: [x_85], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf210 = aten.convolution_backward(buf209, add_96, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_96
    del buf209
    del primals_64
    buf211 = buf210[0]
    buf212 = buf210[1]
    del buf210
    buf214 = buf200; del buf200  # reuse
    buf213 = buf203; del buf203  # reuse
    buf215 = empty((384, ), device='cpu', dtype=torch.float32)
    buf216 = empty((384, ), device='cpu', dtype=torch.float32)
    buf217 = buf205; del buf205  # reuse
    cpp_fused_add_native_batch_norm_backward_39(c_void_p(buf217.data_ptr()), c_void_p(clone_15.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_282.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()))
    del convolution_29
    del primals_62
    del squeeze_40
    del unsqueeze_282
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf218 = aten.convolution_backward(buf217, view_15, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_61
    del view_15
    buf219 = buf218[0]
    buf220 = buf218[1]
    del buf218
    buf221 = reinterpret_tensor(buf214, (48, 196, 64), (64, 3072, 1), 0); del buf214  # reuse
    buf223 = reinterpret_tensor(buf211, (48, 196, 64), (64, 3072, 1), 0); del buf211  # reuse
    cpp_fused_bmm_40(c_void_p(buf219.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf223.data_ptr()))
    buf222 = reinterpret_tensor(buf219, (48, 196, 64), (12544, 64, 1), 0); del buf219  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_72, buf221, out=buf222)
    del permute_72
    buf224 = reinterpret_tensor(buf195, (48, 196, 196), (38416, 196, 1), 0); del buf195  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf223, permute_73, out=buf224)
    del permute_73
    buf225 = buf194; del buf194  # reuse
    buf226 = reinterpret_tensor(buf224, (8, 6, 196, 196), (230496, 38416, 196, 1), 0); del buf224  # reuse
    cpp_fused__softmax_backward_data_mul_41(c_void_p(buf226.data_ptr()), c_void_p(alias_15.data_ptr()), c_void_p(buf225.data_ptr()))
    del alias_15
    buf227 = reinterpret_tensor(buf223, (48, 64, 196), (12544, 196, 1), 0); del buf223  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_74, reinterpret_tensor(buf226, (48, 196, 196), (38416, 196, 1), 0), out=buf227)
    del permute_74
    buf228 = reinterpret_tensor(buf221, (48, 196, 64), (12544, 64, 1), 0); del buf221  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf226, (48, 196, 196), (38416, 196, 1), 0), permute_75, out=buf228)
    del permute_75
    buf229 = buf198; del buf198  # reuse
    cpp_fused_convolution_backward_42(c_void_p(buf228.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf229.data_ptr()))
    del buf222
    del buf227
    del buf228
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf230 = aten.convolution_backward(buf229, add_90, primals_60, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_90
    del primals_60
    buf231 = buf230[0]
    buf232 = buf230[1]
    del buf230
    buf233 = buf215; del buf215  # reuse
    buf234 = empty((384, ), device='cpu', dtype=torch.float32)
    buf235 = buf231; del buf231  # reuse
    buf236 = buf234; del buf234  # reuse
    buf237 = buf217; del buf217  # reuse
    cpp_fused_add_native_batch_norm_backward_43(c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(clone_15.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(unsqueeze_294.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf233.data_ptr()))
    del convolution_27
    del primals_58
    del squeeze_37
    del unsqueeze_294
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf238 = aten.convolution_backward(buf237, clone_22, primals_57, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del clone_22
    del primals_57
    buf239 = buf238[0]
    buf240 = buf238[1]
    del buf238
    buf241 = buf239; del buf239  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_44(c_void_p(buf241.data_ptr()), c_void_p(convolution_26.data_ptr()))
    del convolution_26
    # Source Nodes: [x_73], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf242 = aten.convolution_backward(buf241, add_83, primals_56, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_83
    del buf241
    del primals_56
    buf243 = buf242[0]
    buf244 = buf242[1]
    del buf242
    buf245 = empty((384, ), device='cpu', dtype=torch.float32)
    buf246 = empty((384, ), device='cpu', dtype=torch.float32)
    buf247 = empty((384, ), device='cpu', dtype=torch.float32)
    buf248 = buf237; del buf237  # reuse
    cpp_fused_add_native_batch_norm_backward_45(c_void_p(buf248.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(clone_15.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_306.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()))
    del convolution_25
    del primals_54
    del squeeze_34
    del unsqueeze_306
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf249 = aten.convolution_backward(buf248, view_7, primals_53, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_53
    del view_7
    buf250 = buf249[0]
    buf251 = buf249[1]
    del buf249
    buf252 = reinterpret_tensor(buf243, (48, 196, 64), (64, 3072, 1), 0); del buf243  # reuse
    buf254 = reinterpret_tensor(buf235, (48, 196, 64), (64, 3072, 1), 0); del buf235  # reuse
    cpp_fused_bmm_46(c_void_p(buf250.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf254.data_ptr()))
    buf253 = reinterpret_tensor(buf250, (48, 196, 64), (12544, 64, 1), 0); del buf250  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_79, buf252, out=buf253)
    del permute_79
    buf255 = reinterpret_tensor(buf226, (48, 196, 196), (38416, 196, 1), 0); del buf226  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf254, permute_80, out=buf255)
    del permute_80
    buf256 = buf225; del buf225  # reuse
    buf257 = reinterpret_tensor(buf255, (8, 6, 196, 196), (230496, 38416, 196, 1), 0); del buf255  # reuse
    cpp_fused__softmax_backward_data_mul_47(c_void_p(buf257.data_ptr()), c_void_p(alias_16.data_ptr()), c_void_p(buf256.data_ptr()))
    del alias_16
    del buf256
    buf258 = reinterpret_tensor(buf254, (48, 64, 196), (12544, 196, 1), 0); del buf254  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_81, reinterpret_tensor(buf257, (48, 196, 196), (38416, 196, 1), 0), out=buf258)
    del permute_81
    buf259 = reinterpret_tensor(buf252, (48, 196, 64), (12544, 64, 1), 0); del buf252  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf257, (48, 196, 196), (38416, 196, 1), 0), permute_82, out=buf259)
    del buf257
    del permute_82
    buf260 = buf229; del buf229  # reuse
    cpp_fused_convolution_backward_48(c_void_p(buf259.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf260.data_ptr()))
    del buf253
    del buf258
    del buf259
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf261 = aten.convolution_backward(buf260, add_77, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_77
    del buf260
    del primals_52
    buf262 = buf261[0]
    buf263 = buf261[1]
    del buf261
    buf264 = buf246; del buf246  # reuse
    buf265 = empty((384, ), device='cpu', dtype=torch.float32)
    buf266 = empty((384, ), device='cpu', dtype=torch.float32)
    buf267 = buf248; del buf248  # reuse
    buf268 = empty((1, 384, 14, 14), device='cpu', dtype=torch.float32)
    buf269 = empty((384, ), device='cpu', dtype=torch.float32)
    buf270 = empty((384, ), device='cpu', dtype=torch.float32)
    buf271 = empty((384, ), device='cpu', dtype=torch.float32)
    buf272 = buf267; del buf267  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_sum_49(c_void_p(buf272.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(clone_15.data_ptr()), c_void_p(unsqueeze_318.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_330.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()))
    del buf262
    del buf265
    del buf270
    del clone_15
    del convolution_23
    del primals_48
    del primals_50
    del squeeze_28
    del squeeze_31
    del unsqueeze_318
    del unsqueeze_330
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf273 = aten.convolution_backward(buf272, add_66, primals_46, [384], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_66
    del buf272
    del primals_46
    buf274 = buf273[0]
    buf275 = buf273[1]
    buf276 = buf273[2]
    del buf273
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf277 = aten.convolution_backward(buf274, mul_104, primals_45, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_104
    del primals_45
    buf278 = buf277[0]
    buf279 = buf277[1]
    del buf277
    buf280 = buf278; del buf278  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_50(c_void_p(buf280.data_ptr()), c_void_p(convolution_21.data_ptr()))
    del convolution_21
    # Source Nodes: [x_57], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf281 = aten.convolution_backward(buf280, clone_13, primals_44, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf280
    del clone_13
    del primals_44
    buf282 = buf281[0]
    buf283 = buf281[1]
    del buf281
    buf284 = buf282; del buf282  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_51(c_void_p(buf284.data_ptr()), c_void_p(convolution_20.data_ptr()))
    del convolution_20
    # Source Nodes: [x_54], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf285 = aten.convolution_backward(buf284, add_63, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_63
    del buf284
    del primals_43
    buf286 = buf285[0]
    buf287 = buf285[1]
    del buf285
    buf288 = empty((192, ), device='cpu', dtype=torch.float32)
    buf289 = empty((192, ), device='cpu', dtype=torch.float32)
    buf290 = buf286; del buf286  # reuse
    buf291 = buf289; del buf289  # reuse
    buf292 = buf274; del buf274  # reuse
    cpp_fused_add_native_batch_norm_backward_52(c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_342.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf288.data_ptr()))
    del buf290
    del convolution_19
    del primals_41
    del squeeze_25
    del unsqueeze_342
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf293 = aten.convolution_backward(buf292, mul_91, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_91
    del primals_40
    buf294 = buf293[0]
    buf295 = buf293[1]
    del buf293
    buf296 = buf294; del buf294  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_53(c_void_p(buf296.data_ptr()), c_void_p(convolution_18.data_ptr()))
    del convolution_18
    # Source Nodes: [x_49], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf297 = aten.convolution_backward(buf296, clone_11, primals_39, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf296
    del clone_11
    del primals_39
    buf298 = buf297[0]
    buf299 = buf297[1]
    del buf297
    buf300 = buf298; del buf298  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_54(c_void_p(buf300.data_ptr()), c_void_p(convolution_17.data_ptr()))
    del convolution_17
    # Source Nodes: [x_46], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf301 = aten.convolution_backward(buf300, add_55, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_55
    del buf300
    del primals_38
    buf302 = buf301[0]
    buf303 = buf301[1]
    del buf301
    buf304 = empty((192, ), device='cpu', dtype=torch.float32)
    buf305 = empty((192, ), device='cpu', dtype=torch.float32)
    buf306 = empty((192, ), device='cpu', dtype=torch.float32)
    buf307 = buf292; del buf292  # reuse
    cpp_fused_add_native_batch_norm_backward_55(c_void_p(buf307.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_354.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()))
    del buf302
    del convolution_16
    del primals_36
    del squeeze_22
    del unsqueeze_354
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf308 = aten.convolution_backward(buf307, mul_78, primals_35, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_78
    del primals_35
    buf309 = buf308[0]
    buf310 = buf308[1]
    del buf308
    buf311 = buf309; del buf309  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_56(c_void_p(buf311.data_ptr()), c_void_p(convolution_15.data_ptr()))
    del convolution_15
    # Source Nodes: [x_41], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf312 = aten.convolution_backward(buf311, clone_9, primals_34, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf311
    del clone_9
    del primals_34
    buf313 = buf312[0]
    buf314 = buf312[1]
    del buf312
    buf315 = buf313; del buf313  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_57(c_void_p(buf315.data_ptr()), c_void_p(convolution_14.data_ptr()))
    del convolution_14
    # Source Nodes: [x_38], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf316 = aten.convolution_backward(buf315, add_47, primals_33, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_47
    del buf315
    del primals_33
    buf317 = buf316[0]
    buf318 = buf316[1]
    del buf316
    buf319 = buf305; del buf305  # reuse
    buf320 = empty((192, ), device='cpu', dtype=torch.float32)
    buf321 = empty((192, ), device='cpu', dtype=torch.float32)
    buf322 = buf0; del buf0  # reuse
    cpp_fused_add_native_batch_norm_backward_58(c_void_p(buf322.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(unsqueeze_366.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()))
    del buf307
    del primals_31
    del squeeze_19
    del unsqueeze_366
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf323 = aten.convolution_backward(buf322, mul_65, primals_30, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_65
    del primals_30
    buf324 = buf323[0]
    buf325 = buf323[1]
    del buf323
    buf326 = buf324; del buf324  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_59(c_void_p(buf326.data_ptr()), c_void_p(convolution_12.data_ptr()))
    del convolution_12
    # Source Nodes: [x_33], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf327 = aten.convolution_backward(buf326, clone_7, primals_29, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf326
    del clone_7
    del primals_29
    buf328 = buf327[0]
    buf329 = buf327[1]
    del buf327
    buf330 = buf328; del buf328  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_60(c_void_p(buf330.data_ptr()), c_void_p(convolution_11.data_ptr()))
    del convolution_11
    # Source Nodes: [x_30], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf331 = aten.convolution_backward(buf330, add_39, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_39
    del buf330
    del primals_28
    buf332 = buf331[0]
    buf333 = buf331[1]
    del buf331
    buf335 = buf317; del buf317  # reuse
    buf334 = buf320; del buf320  # reuse
    buf336 = empty((192, ), device='cpu', dtype=torch.float32)
    buf337 = empty((192, ), device='cpu', dtype=torch.float32)
    buf338 = buf322; del buf322  # reuse
    cpp_fused_add_native_batch_norm_backward_61(c_void_p(buf338.data_ptr()), c_void_p(clone.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_378.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()))
    del buf332
    del buf335
    del convolution_10
    del primals_26
    del squeeze_16
    del unsqueeze_378
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf339 = aten.convolution_backward(buf338, mul_52, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_52
    del primals_25
    buf340 = buf339[0]
    buf341 = buf339[1]
    del buf339
    buf342 = buf340; del buf340  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_62(c_void_p(buf342.data_ptr()), c_void_p(convolution_9.data_ptr()))
    del convolution_9
    # Source Nodes: [x_25], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf343 = aten.convolution_backward(buf342, clone_5, primals_24, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf342
    del clone_5
    del primals_24
    buf344 = buf343[0]
    buf345 = buf343[1]
    del buf343
    buf346 = buf344; del buf344  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_63(c_void_p(buf346.data_ptr()), c_void_p(convolution_8.data_ptr()))
    del convolution_8
    # Source Nodes: [x_22], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf347 = aten.convolution_backward(buf346, add_31, primals_23, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_31
    del buf346
    del primals_23
    buf348 = buf347[0]
    buf349 = buf347[1]
    del buf347
    buf350 = buf336; del buf336  # reuse
    buf351 = empty((192, ), device='cpu', dtype=torch.float32)
    buf352 = buf348; del buf348  # reuse
    buf353 = buf351; del buf351  # reuse
    buf354 = buf338; del buf338  # reuse
    cpp_fused_add_native_batch_norm_backward_64(c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(clone.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_390.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf350.data_ptr()))
    del buf352
    del convolution_7
    del primals_21
    del squeeze_13
    del unsqueeze_390
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf355 = aten.convolution_backward(buf354, mul_39, primals_20, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_39
    del primals_20
    buf356 = buf355[0]
    buf357 = buf355[1]
    del buf355
    buf358 = buf356; del buf356  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_65(c_void_p(buf358.data_ptr()), c_void_p(convolution_6.data_ptr()))
    del convolution_6
    # Source Nodes: [x_17], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf359 = aten.convolution_backward(buf358, clone_3, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf358
    del clone_3
    del primals_19
    buf360 = buf359[0]
    buf361 = buf359[1]
    del buf359
    buf362 = buf360; del buf360  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_66(c_void_p(buf362.data_ptr()), c_void_p(convolution_5.data_ptr()))
    del convolution_5
    # Source Nodes: [x_14], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf363 = aten.convolution_backward(buf362, add_23, primals_18, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_23
    del buf362
    del primals_18
    buf364 = buf363[0]
    buf365 = buf363[1]
    del buf363
    buf366 = empty((192, ), device='cpu', dtype=torch.float32)
    buf367 = empty((192, ), device='cpu', dtype=torch.float32)
    buf368 = empty((192, ), device='cpu', dtype=torch.float32)
    buf369 = buf354; del buf354  # reuse
    cpp_fused_add_native_batch_norm_backward_67(c_void_p(buf369.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(clone.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_402.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()))
    del buf364
    del convolution_4
    del primals_16
    del squeeze_10
    del unsqueeze_402
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf370 = aten.convolution_backward(buf369, mul_26, primals_15, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_26
    del primals_15
    buf371 = buf370[0]
    buf372 = buf370[1]
    del buf370
    buf373 = buf371; del buf371  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_68(c_void_p(buf373.data_ptr()), c_void_p(convolution_3.data_ptr()))
    del convolution_3
    # Source Nodes: [x_9], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf374 = aten.convolution_backward(buf373, clone_1, primals_14, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf373
    del clone_1
    del primals_14
    buf375 = buf374[0]
    buf376 = buf374[1]
    del buf374
    buf377 = buf375; del buf375  # reuse
    cpp_fused_convolution_backward_gelu_gelu_backward_69(c_void_p(buf377.data_ptr()), c_void_p(convolution_2.data_ptr()))
    del convolution_2
    # Source Nodes: [x_6], Original ATen: [aten.convolution_backward, aten.gelu, aten.gelu_backward]
    buf378 = aten.convolution_backward(buf377, add_15, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_15
    del buf377
    del primals_13
    buf379 = buf378[0]
    buf380 = buf378[1]
    del buf378
    buf381 = buf367; del buf367  # reuse
    buf382 = empty((192, ), device='cpu', dtype=torch.float32)
    buf383 = empty((192, ), device='cpu', dtype=torch.float32)
    buf384 = buf369; del buf369  # reuse
    buf385 = empty((1, 192, 28, 28), device='cpu', dtype=torch.float32)
    buf386 = empty((192, ), device='cpu', dtype=torch.float32)
    buf387 = empty((192, ), device='cpu', dtype=torch.float32)
    buf388 = empty((192, ), device='cpu', dtype=torch.float32)
    buf389 = buf384; del buf384  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_sum_70(c_void_p(buf389.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(clone.data_ptr()), c_void_p(unsqueeze_414.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_426.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf388.data_ptr()))
    del buf379
    del buf382
    del buf387
    del clone
    del convolution_1
    del primals_11
    del primals_9
    del squeeze_4
    del squeeze_7
    del unsqueeze_414
    del unsqueeze_426
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf390 = aten.convolution_backward(buf389, relu, primals_7, [192], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf389
    del primals_7
    buf391 = buf390[0]
    buf392 = buf390[1]
    buf393 = buf390[2]
    del buf390
    buf394 = empty((32, ), device='cpu', dtype=torch.float32)
    buf395 = empty((32, ), device='cpu', dtype=torch.float32)
    buf396 = empty((32, ), device='cpu', dtype=torch.float32)
    buf397 = buf391; del buf391  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_71(c_void_p(buf397.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_438.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()))
    del buf395
    del convolution
    del primals_5
    del relu
    del squeeze_1
    del unsqueeze_438
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf398 = aten.convolution_backward(buf397, primals_206, primals_4, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf397
    del primals_206
    del primals_4
    buf399 = buf398[1]
    return (buf385, buf268, buf135, buf399, buf396, buf394, buf392, buf393, buf388, buf386, buf383, buf381, buf380, buf376, buf372, buf368, buf366, buf365, buf361, buf357, buf353, buf350, buf349, buf345, buf341, buf337, buf334, buf333, buf329, buf325, buf321, buf319, buf318, buf314, buf310, buf306, buf304, buf303, buf299, buf295, buf291, buf288, buf287, buf283, buf279, buf275, buf276, buf271, buf269, buf266, buf264, buf263, buf251, buf247, buf245, buf244, buf240, buf236, buf233, buf232, buf220, buf216, buf213, buf212, buf208, buf204, buf202, buf201, buf189, buf185, buf183, buf182, buf178, buf174, buf171, buf170, buf158, buf154, buf151, buf150, buf146, buf142, buf143, buf138, buf136, buf133, buf131, buf130, buf118, buf114, buf112, buf111, buf107, buf103, buf100, buf99, buf87, buf83, buf80, buf79, buf75, buf71, buf69, buf68, buf56, buf52, buf50, buf49, buf45, buf41, buf38, buf37, buf25, buf21, buf18, buf17, buf13, buf10, buf6, reinterpret_tensor(buf4, (1000, 768), (768, 1), 0), reinterpret_tensor(buf5, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((32, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((192, 32, 4, 4), (512, 1, 128, 32), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((384, 48, 3, 3), (432, 1, 144, 48), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((384, 48, 3, 3), (432, 1, 144, 48), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((384, 48, 3, 3), (432, 1, 144, 48), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((384, 48, 3, 3), (432, 1, 144, 48), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((384, 48, 3, 3), (432, 1, 144, 48), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((384, 48, 3, 3), (432, 1, 144, 48), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((384, 48, 3, 3), (432, 1, 144, 48), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((384, 192, 2, 2), (768, 1, 384, 192), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((1152, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((768, 384, 2, 2), (1536, 1, 768, 384), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((2304, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((3072, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((768, 3072, 1, 1), (3072, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    relu = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    clone = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    add_15 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    clone_1 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    mul_26 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    add_23 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    clone_3 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    mul_39 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    add_31 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    clone_5 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    mul_52 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    add_39 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    clone_7 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    mul_65 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    add_47 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    clone_9 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    mul_78 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    add_55 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    clone_11 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    mul_91 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    add_63 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    clone_13 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    mul_104 = rand_strided((8, 384, 28, 28), (301056, 1, 10752, 384), device='cpu', dtype=torch.float32)
    add_66 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    clone_15 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    add_77 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    view_7 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    add_83 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    clone_22 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    add_90 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    view_15 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    add_96 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    clone_30 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    add_103 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    view_23 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    add_109 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    clone_38 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    add_116 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    view_31 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    add_122 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    clone_46 = rand_strided((8, 1536, 14, 14), (301056, 1, 21504, 1536), device='cpu', dtype=torch.float32)
    add_124 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    clone_48 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_135 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    view_39 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_141 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cpu', dtype=torch.float32)
    clone_55 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_148 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    view_47 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_154 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cpu', dtype=torch.float32)
    clone_63 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_161 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    view_55 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_167 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cpu', dtype=torch.float32)
    clone_71 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_174 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    view_63 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    convolution_54 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_180 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cpu', dtype=torch.float32)
    clone_79 = rand_strided((8, 3072, 7, 7), (150528, 1, 21504, 3072), device='cpu', dtype=torch.float32)
    convolution_56 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    clone_81 = rand_strided((8, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_25 = rand_strided((1000, 768), (768, 1), device='cpu', dtype=torch.float32)
    unsqueeze_114 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_126 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    permute_30 = rand_strided((48, 49, 49), (2401, 1, 49), device='cpu', dtype=torch.float32)
    permute_31 = rand_strided((48, 128, 49), (6272, 1, 128), device='cpu', dtype=torch.float32)
    alias_9 = rand_strided((8, 6, 49, 49), (14406, 1, 294, 6), device='cpu', dtype=torch.float32)
    permute_32 = rand_strided((48, 128, 49), (6272, 1, 128), device='cpu', dtype=torch.float32)
    permute_33 = rand_strided((48, 49, 128), (6272, 1, 49), device='cpu', dtype=torch.float32)
    unsqueeze_138 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_150 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    permute_37 = rand_strided((48, 49, 49), (2401, 1, 49), device='cpu', dtype=torch.float32)
    permute_38 = rand_strided((48, 128, 49), (6272, 1, 128), device='cpu', dtype=torch.float32)
    alias_10 = rand_strided((8, 6, 49, 49), (14406, 1, 294, 6), device='cpu', dtype=torch.float32)
    permute_39 = rand_strided((48, 128, 49), (6272, 1, 128), device='cpu', dtype=torch.float32)
    permute_40 = rand_strided((48, 49, 128), (6272, 1, 49), device='cpu', dtype=torch.float32)
    unsqueeze_162 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_174 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    permute_44 = rand_strided((48, 49, 49), (2401, 1, 49), device='cpu', dtype=torch.float32)
    permute_45 = rand_strided((48, 128, 49), (6272, 1, 128), device='cpu', dtype=torch.float32)
    alias_11 = rand_strided((8, 6, 49, 49), (14406, 1, 294, 6), device='cpu', dtype=torch.float32)
    permute_46 = rand_strided((48, 128, 49), (6272, 1, 128), device='cpu', dtype=torch.float32)
    permute_47 = rand_strided((48, 49, 128), (6272, 1, 49), device='cpu', dtype=torch.float32)
    unsqueeze_186 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_198 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    permute_51 = rand_strided((48, 49, 49), (2401, 1, 49), device='cpu', dtype=torch.float32)
    permute_52 = rand_strided((48, 128, 49), (6272, 1, 128), device='cpu', dtype=torch.float32)
    alias_12 = rand_strided((8, 6, 49, 49), (14406, 1, 294, 6), device='cpu', dtype=torch.float32)
    permute_53 = rand_strided((48, 128, 49), (6272, 1, 128), device='cpu', dtype=torch.float32)
    permute_54 = rand_strided((48, 49, 128), (6272, 1, 49), device='cpu', dtype=torch.float32)
    unsqueeze_210 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_222 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_234 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    permute_58 = rand_strided((48, 196, 196), (38416, 1, 196), device='cpu', dtype=torch.float32)
    permute_59 = rand_strided((48, 64, 196), (12544, 1, 64), device='cpu', dtype=torch.float32)
    alias_13 = rand_strided((8, 6, 196, 196), (230496, 1, 1176, 6), device='cpu', dtype=torch.float32)
    permute_60 = rand_strided((48, 64, 196), (12544, 1, 64), device='cpu', dtype=torch.float32)
    permute_61 = rand_strided((48, 196, 64), (12544, 1, 196), device='cpu', dtype=torch.float32)
    unsqueeze_246 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    permute_65 = rand_strided((48, 196, 196), (38416, 1, 196), device='cpu', dtype=torch.float32)
    permute_66 = rand_strided((48, 64, 196), (12544, 1, 64), device='cpu', dtype=torch.float32)
    alias_14 = rand_strided((8, 6, 196, 196), (230496, 1, 1176, 6), device='cpu', dtype=torch.float32)
    permute_67 = rand_strided((48, 64, 196), (12544, 1, 64), device='cpu', dtype=torch.float32)
    permute_68 = rand_strided((48, 196, 64), (12544, 1, 196), device='cpu', dtype=torch.float32)
    unsqueeze_270 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_282 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    permute_72 = rand_strided((48, 196, 196), (38416, 1, 196), device='cpu', dtype=torch.float32)
    permute_73 = rand_strided((48, 64, 196), (12544, 1, 64), device='cpu', dtype=torch.float32)
    alias_15 = rand_strided((8, 6, 196, 196), (230496, 1, 1176, 6), device='cpu', dtype=torch.float32)
    permute_74 = rand_strided((48, 64, 196), (12544, 1, 64), device='cpu', dtype=torch.float32)
    permute_75 = rand_strided((48, 196, 64), (12544, 1, 196), device='cpu', dtype=torch.float32)
    unsqueeze_294 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_306 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    permute_79 = rand_strided((48, 196, 196), (38416, 1, 196), device='cpu', dtype=torch.float32)
    permute_80 = rand_strided((48, 64, 196), (12544, 1, 64), device='cpu', dtype=torch.float32)
    alias_16 = rand_strided((8, 6, 196, 196), (230496, 1, 1176, 6), device='cpu', dtype=torch.float32)
    permute_81 = rand_strided((48, 64, 196), (12544, 1, 64), device='cpu', dtype=torch.float32)
    permute_82 = rand_strided((48, 196, 64), (12544, 1, 196), device='cpu', dtype=torch.float32)
    unsqueeze_318 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_342 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_354 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_366 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_378 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_390 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_414 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_426 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_438 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_5, primals_7, primals_9, primals_11, primals_13, primals_14, primals_15, primals_16, primals_18, primals_19, primals_20, primals_21, primals_23, primals_24, primals_25, primals_26, primals_28, primals_29, primals_30, primals_31, primals_33, primals_34, primals_35, primals_36, primals_38, primals_39, primals_40, primals_41, primals_43, primals_44, primals_45, primals_46, primals_48, primals_50, primals_52, primals_53, primals_54, primals_56, primals_57, primals_58, primals_60, primals_61, primals_62, primals_64, primals_65, primals_66, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_76, primals_77, primals_78, primals_80, primals_81, primals_82, primals_84, primals_86, primals_88, primals_89, primals_90, primals_92, primals_93, primals_94, primals_96, primals_97, primals_98, primals_100, primals_101, primals_102, primals_104, primals_105, primals_106, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_116, primals_117, primals_118, primals_206, convolution, squeeze_1, relu, convolution_1, squeeze_4, clone, squeeze_7, add_15, convolution_2, clone_1, convolution_3, mul_26, convolution_4, squeeze_10, add_23, convolution_5, clone_3, convolution_6, mul_39, convolution_7, squeeze_13, add_31, convolution_8, clone_5, convolution_9, mul_52, convolution_10, squeeze_16, add_39, convolution_11, clone_7, convolution_12, mul_65, convolution_13, squeeze_19, add_47, convolution_14, clone_9, convolution_15, mul_78, convolution_16, squeeze_22, add_55, convolution_17, clone_11, convolution_18, mul_91, convolution_19, squeeze_25, add_63, convolution_20, clone_13, convolution_21, mul_104, add_66, convolution_23, squeeze_28, clone_15, squeeze_31, add_77, view_7, convolution_25, squeeze_34, add_83, convolution_26, clone_22, convolution_27, squeeze_37, add_90, view_15, convolution_29, squeeze_40, add_96, convolution_30, clone_30, convolution_31, squeeze_43, add_103, view_23, convolution_33, squeeze_46, add_109, convolution_34, clone_38, convolution_35, squeeze_49, add_116, view_31, convolution_37, squeeze_52, add_122, convolution_38, clone_46, add_124, convolution_40, squeeze_55, clone_48, squeeze_58, add_135, view_39, convolution_42, squeeze_61, add_141, convolution_43, clone_55, convolution_44, squeeze_64, add_148, view_47, convolution_46, squeeze_67, add_154, convolution_47, clone_63, convolution_48, squeeze_70, add_161, view_55, convolution_50, squeeze_73, add_167, convolution_51, clone_71, convolution_52, squeeze_76, add_174, view_63, convolution_54, squeeze_79, add_180, convolution_55, clone_79, convolution_56, squeeze_82, clone_81, permute_25, unsqueeze_114, unsqueeze_126, permute_30, permute_31, alias_9, permute_32, permute_33, unsqueeze_138, unsqueeze_150, permute_37, permute_38, alias_10, permute_39, permute_40, unsqueeze_162, unsqueeze_174, permute_44, permute_45, alias_11, permute_46, permute_47, unsqueeze_186, unsqueeze_198, permute_51, permute_52, alias_12, permute_53, permute_54, unsqueeze_210, unsqueeze_222, unsqueeze_234, permute_58, permute_59, alias_13, permute_60, permute_61, unsqueeze_246, unsqueeze_258, permute_65, permute_66, alias_14, permute_67, permute_68, unsqueeze_270, unsqueeze_282, permute_72, permute_73, alias_15, permute_74, permute_75, unsqueeze_294, unsqueeze_306, permute_79, permute_80, alias_16, permute_81, permute_82, unsqueeze_318, unsqueeze_330, unsqueeze_342, unsqueeze_354, unsqueeze_366, unsqueeze_378, unsqueeze_390, unsqueeze_402, unsqueeze_414, unsqueeze_426, unsqueeze_438, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('visformer_small', benchmark_compiled_module)
