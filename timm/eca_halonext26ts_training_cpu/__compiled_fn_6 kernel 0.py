
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x2) + (131072L*x1)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x2) + (131072L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(64.0);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2048L); x2+=static_cast<long>(8L))
                    {
                        float tmp24[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (2048L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (2048L*x1) + (2048L*x1_inner) + (131072L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                            auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                            auto tmp1 = static_cast<float>(64.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = tmp3 * tmp4;
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
                            tmp23.store(tmp24 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp24, 8, out_ptr4 + static_cast<long>(x1 + (64L*x2) + (131072L*x0)), static_cast<long>(64L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_native_batch_norm_backward_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp19[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (32768L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (32768L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (32768L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp5 = static_cast<float>(0.001953125);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = tmp8 * tmp8;
                            auto tmp10 = tmp7 * tmp9;
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp12 = tmp2 - tmp11;
                            auto tmp14 = tmp13 * tmp6;
                            auto tmp15 = tmp12 - tmp14;
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp18 = tmp15 * tmp17;
                            tmp18.store(tmp19 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp19, 8, out_ptr3 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)), static_cast<long>(64L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_view_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (144L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (144L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(12L); x3+=static_cast<long>(1L))
                        {
                            {
                                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                                float tmp_acc0 = 0;
                                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                                for(long x4=static_cast<long>(0L); x4<static_cast<long>(8L); x4+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x4 + (12L*x3) + (144L*x2) + (1152L*x1) + (9216L*x0)));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x4 + (12L*x3) + (144L*x2) + (1152L*x1) + (9216L*x0)));
                                    auto tmp3 = out_ptr0[static_cast<long>(x2 + (8L*x1) + (64L*x0))];
                                    auto tmp2 = tmp0 * tmp1;
                                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                    auto tmp5 = tmp1 * tmp4;
                                    auto tmp6 = tmp2 - tmp5;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                }
                                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                                for(long x4=static_cast<long>(8L); x4<static_cast<long>(12L); x4+=static_cast<long>(1L))
                                {
                                    auto tmp0 = in_ptr0[static_cast<long>(x4 + (12L*x3) + (144L*x2) + (1152L*x1) + (9216L*x0))];
                                    auto tmp1 = in_ptr1[static_cast<long>(x4 + (12L*x3) + (144L*x2) + (1152L*x1) + (9216L*x0))];
                                    auto tmp3 = out_ptr0[static_cast<long>(x2 + (8L*x1) + (64L*x0))];
                                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                    tmp_acc0 = tmp_acc0 + tmp5;
                                }
                                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                                out_ptr1[static_cast<long>(x3 + (12L*x1) + (96L*x2) + (768L*x0))] = static_cast<float>(tmp_acc0);
                            }
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(23L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(24);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L))));
                            auto tmp5 = static_cast<long>(207);
                            auto tmp6 = tmp4 < tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L)))), 23L)) % static_cast<long>(9L));
                                auto tmp9 = static_cast<long>(8);
                                auto tmp10 = tmp8 < tmp9;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L))))) % static_cast<long>(23L));
                                    auto tmp13 = static_cast<long>(11);
                                    auto tmp14 = tmp12 >= tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp16 = out_ptr1[static_cast<long>((-11L) + (12L*(static_cast<long>(c10::div_floor_integer((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L)))), 23L)) % static_cast<long>(9L))) + (96L*(c10::div_floor_integer(x0, 8L))) + (static_cast<long>((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L))))) % static_cast<long>(23L)))];
                                        return tmp16;
                                    }
                                    ;
                                    auto tmp17 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                                    auto tmp18 = static_cast<float>(0.0);
                                    auto tmp19 = tmp14 ? tmp17 : tmp18;
                                    return tmp19;
                                }
                                ;
                                auto tmp20 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                                auto tmp21 = static_cast<float>(0.0);
                                auto tmp22 = tmp10 ? tmp20 : tmp21;
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr2[static_cast<long>(x1 + (23L*x0))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (12L*x2) + (144L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (12L*x2) + (144L*x0)));
                            auto tmp3 = in_ptr2[static_cast<long>(x0)];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (12L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (12L*x2) + (144L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x2) + (144L*x0))];
                            auto tmp3 = in_ptr2[static_cast<long>(x0)];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            tmp_acc0 = tmp_acc0 + tmp5;
                        }
                        out_ptr0[static_cast<long>(x1 + (12L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(23L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(24);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L))));
                            auto tmp5 = static_cast<long>(207);
                            auto tmp6 = tmp4 < tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L)))), 23L)) % static_cast<long>(9L));
                                auto tmp9 = static_cast<long>(8);
                                auto tmp10 = tmp8 < tmp9;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L))))) % static_cast<long>(23L));
                                    auto tmp13 = static_cast<long>(11);
                                    auto tmp14 = tmp12 >= tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp16 = out_ptr0[static_cast<long>((-11L) + (12L*(static_cast<long>(c10::div_floor_integer((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L)))), 23L)) % static_cast<long>(9L))) + (96L*(c10::div_floor_integer(x0, 8L))) + (static_cast<long>((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L))))) % static_cast<long>(23L)))];
                                        return tmp16;
                                    }
                                    ;
                                    auto tmp17 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                                    auto tmp18 = static_cast<float>(0.0);
                                    auto tmp19 = tmp14 ? tmp17 : tmp18;
                                    return tmp19;
                                }
                                ;
                                auto tmp20 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                                auto tmp21 = static_cast<float>(0.0);
                                auto tmp22 = tmp10 ? tmp20 : tmp21;
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr1[static_cast<long>(x1 + (23L*x0))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    auto tmp3 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(0.25);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_unfold_backward_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       int* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(737280L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(144L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(16);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (144L*x1) + (2304L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(80);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>((-16L) + x1 + (64L*x2) + (9216L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp14 = tmp4 ? tmp7 : tmp13;
                        out_ptr1[static_cast<long>(x2 + (144L*x1) + (11520L*x0))] = tmp14;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(5120L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (144L*x0)));
                    tmp0.store(in_out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<int>(x0);
                    out_ptr2[static_cast<long>(x0)] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_unfold_backward_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(737280L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_constant_pad_nd_convolution_backward_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(5120L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(2L + x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(12);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = c10::convert<long>(2L + x2);
                        auto tmp6 = tmp5 >= tmp1;
                        auto tmp7 = tmp5 < tmp3;
                        auto tmp8 = tmp2 & tmp4;
                        auto tmp9 = tmp8 & tmp6;
                        auto tmp10 = tmp9 & tmp7;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr0[static_cast<long>(26L + x2 + (12L*x1) + (144L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        out_ptr0[static_cast<long>(x2 + (8L*x1) + (64L*x0))] = tmp13;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((16L*x1) + (128L*x2) + (128L*x2_inner) + (1024L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 1024L))) + (8192L*x0) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 64L)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((16L*x2) + (16L*x2_inner) + (128L*x1) + (1024L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 1024L))) + (8192L*x0) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 64L)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((16L*x2) + (16L*x2_inner) + (128L*x1) + (1024L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 1024L))) + (8192L*x0) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 64L)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp4.store(out_ptr0 + static_cast<long>(x2 + (8L*x1) + (64L*x3) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_batch_norm_backward_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(0.001953125);
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_10 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2048L*x2) + (131072L*x1)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2048L*x2) + (131072L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2048L*x2) + (131072L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2048L*x2) + (131072L*x1)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (2048L*x2) + (131072L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(64.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp7 = tmp5 + tmp6;
                            auto tmp9 = tmp7 * tmp8;
                            auto tmp12 = tmp10 - tmp11;
                            auto tmp13 = tmp9 * tmp12;
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp9 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp9;
                            tmp_acc1_vec = tmp_acc1_vec + tmp13;
                            tmp_acc2_vec = tmp_acc2_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2048L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (2048L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (2048L*x1) + (131072L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (2048L*x1) + (131072L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (2048L*x1) + (131072L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (2048L*x1) + (131072L*x0)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2));
                        auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (2048L*x1) + (131072L*x0)));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp28 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(64.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp12 = tmp10 - tmp11;
                        auto tmp14 = static_cast<float>(0.001953125);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp18 = tmp17 * tmp17;
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp20 = tmp12 * tmp19;
                        auto tmp21 = tmp9 - tmp20;
                        auto tmp23 = tmp22 * tmp15;
                        auto tmp24 = tmp21 - tmp23;
                        auto tmp27 = tmp25 - tmp26;
                        auto tmp29 = tmp28 * tmp15;
                        auto tmp31 = tmp30 * tmp30;
                        auto tmp32 = tmp29 * tmp31;
                        auto tmp33 = tmp27 * tmp32;
                        auto tmp34 = tmp9 - tmp33;
                        auto tmp35 = tmp34 - tmp23;
                        tmp24.store(out_ptr3 + static_cast<long>(x2 + (2048L*x1) + (131072L*x0)));
                        tmp35.store(out_ptr4 + static_cast<long>(x2 + (2048L*x1) + (131072L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr3 + static_cast<long>(x1 + (2048L*x2) + (131072L*x0)), static_cast<long>(2048L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = in_ptr10[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            tmp6.store(out_ptr5 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (131072L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (2048L*x2) + (131072L*x0)), static_cast<long>(2048L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr0[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            tmp6.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (131072L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_native_batch_norm_backward_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>((512L*(static_cast<long>(x1) % static_cast<long>(4L))) + (2048L*(static_cast<long>(x0) % static_cast<long>(2L))) + (4096L*(c10::div_floor_integer(x1, 4L))) + (16384L*(c10::div_floor_integer(x0, 2L))) + (32768L*(c10::div_floor_integer((x3 + x3_inner + (64L*x2)), 512L))) + (static_cast<long>((x3 + x3_inner + (64L*x2))) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>((512L*(static_cast<long>(x1) % static_cast<long>(4L))) + (2048L*(static_cast<long>(x0) % static_cast<long>(2L))) + (4096L*(c10::div_floor_integer(x1, 4L))) + (16384L*(c10::div_floor_integer(x0, 2L))) + (32768L*(c10::div_floor_integer((x3 + x3_inner + (64L*x2)), 512L))) + (static_cast<long>((x3 + x3_inner + (64L*x2))) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>((512L*(static_cast<long>(x1) % static_cast<long>(4L))) + (2048L*(static_cast<long>(x0) % static_cast<long>(2L))) + (4096L*(c10::div_floor_integer(x1, 4L))) + (16384L*(c10::div_floor_integer(x0, 2L))) + (32768L*(c10::div_floor_integer((x3 + x3_inner + (64L*x2)), 512L))) + (static_cast<long>((x3 + x3_inner + (64L*x2))) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = out_ptr1[static_cast<long>(static_cast<long>((x3 + x3_inner + (64L*x2))) % static_cast<long>(512L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp8 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr3[static_cast<long>(static_cast<long>((x3 + x3_inner + (64L*x2))) % static_cast<long>(512L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp13 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = out_ptr0[static_cast<long>(static_cast<long>((x3 + x3_inner + (64L*x2))) % static_cast<long>(512L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp16 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr4[static_cast<long>(static_cast<long>((x3 + x3_inner + (64L*x2))) % static_cast<long>(512L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp5 = static_cast<float>(0.001953125);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = tmp8 * tmp8;
                            auto tmp10 = tmp7 * tmp9;
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp12 = tmp2 - tmp11;
                            auto tmp14 = tmp13 * tmp6;
                            auto tmp15 = tmp12 - tmp14;
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp18 = tmp15 * tmp17;
                            tmp18.store(out_ptr3 + static_cast<long>(x3 + (64L*x1) + (1024L*x0) + (4096L*x2)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_view_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(144L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (144L*x2) + (2304L*x1) + (9216L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (4L*x3) + (4L*x3_inner) + (576L*x2) + (9216L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (16L*x1) + (64L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(12L); x3+=static_cast<long>(1L))
                        {
                            {
                                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                                float tmp_acc0 = 0;
                                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                                for(long x4=static_cast<long>(0L); x4<static_cast<long>(8L); x4+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x4 + (12L*x3) + (144L*x2) + (576L*x1) + (2304L*x0)));
                                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x4_inner = 0; x4_inner < 8; x4_inner++) tmpbuf[x4_inner] = in_ptr1[static_cast<long>((4L*x4) + (4L*x4_inner) + (48L*x3) + (576L*x2) + (2304L*x1) + (9216L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                    auto tmp3 = out_ptr0[static_cast<long>(x2 + (4L*x1) + (16L*x0))];
                                    auto tmp2 = tmp0 * tmp1;
                                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                    auto tmp5 = tmp1 * tmp4;
                                    auto tmp6 = tmp2 - tmp5;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                }
                                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                                for(long x4=static_cast<long>(8L); x4<static_cast<long>(12L); x4+=static_cast<long>(1L))
                                {
                                    auto tmp0 = in_ptr0[static_cast<long>(x4 + (12L*x3) + (144L*x2) + (576L*x1) + (2304L*x0))];
                                    auto tmp1 = in_ptr1[static_cast<long>((4L*x4) + (48L*x3) + (576L*x2) + (2304L*x1) + (9216L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L)))];
                                    auto tmp3 = out_ptr0[static_cast<long>(x2 + (4L*x1) + (16L*x0))];
                                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                    tmp_acc0 = tmp_acc0 + tmp5;
                                }
                                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                                out_ptr1[static_cast<long>(x3 + (12L*x1) + (48L*x2) + (192L*x0))] = static_cast<float>(tmp_acc0);
                            }
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(23L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(24);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x1 + (24L*(static_cast<long>(x0) % static_cast<long>(4L))));
                            auto tmp5 = static_cast<long>(115);
                            auto tmp6 = tmp4 < tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(4L)))), 23L)) % static_cast<long>(5L));
                                auto tmp9 = static_cast<long>(4);
                                auto tmp10 = tmp8 < tmp9;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(4L))))) % static_cast<long>(23L));
                                    auto tmp13 = static_cast<long>(11);
                                    auto tmp14 = tmp12 >= tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp16 = out_ptr1[static_cast<long>((-11L) + (12L*(static_cast<long>(c10::div_floor_integer((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(4L)))), 23L)) % static_cast<long>(5L))) + (48L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(4L))))) % static_cast<long>(23L)))];
                                        return tmp16;
                                    }
                                    ;
                                    auto tmp17 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                                    auto tmp18 = static_cast<float>(0.0);
                                    auto tmp19 = tmp14 ? tmp17 : tmp18;
                                    return tmp19;
                                }
                                ;
                                auto tmp20 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                                auto tmp21 = static_cast<float>(0.0);
                                auto tmp22 = tmp10 ? tmp20 : tmp21;
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr2[static_cast<long>(x1 + (23L*x0))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(12L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (12L*x3) + (144L*x1) + (2304L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((4L*x2) + (4L*x2_inner) + (48L*x3) + (576L*x1) + (9216L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = in_ptr2[static_cast<long>(x1 + (16L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x2 + (12L*x1) + (192L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(12L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (12L*x3) + (144L*x1) + (2304L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>((4L*x2) + (48L*x3) + (576L*x1) + (9216L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L)))];
                                auto tmp3 = in_ptr2[static_cast<long>(x1 + (16L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr0[static_cast<long>(x2 + (12L*x1) + (192L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(23L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(24);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x1 + (24L*(static_cast<long>(x0) % static_cast<long>(4L))));
                            auto tmp5 = static_cast<long>(115);
                            auto tmp6 = tmp4 < tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(4L)))), 23L)) % static_cast<long>(5L));
                                auto tmp9 = static_cast<long>(4);
                                auto tmp10 = tmp8 < tmp9;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(4L))))) % static_cast<long>(23L));
                                    auto tmp13 = static_cast<long>(11);
                                    auto tmp14 = tmp12 >= tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp16 = out_ptr0[static_cast<long>((-11L) + (12L*(static_cast<long>(c10::div_floor_integer((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(4L)))), 23L)) % static_cast<long>(5L))) + (48L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(4L))))) % static_cast<long>(23L)))];
                                        return tmp16;
                                    }
                                    ;
                                    auto tmp17 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                                    auto tmp18 = static_cast<float>(0.0);
                                    auto tmp19 = tmp14 ? tmp17 : tmp18;
                                    return tmp19;
                                }
                                ;
                                auto tmp20 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                                auto tmp21 = static_cast<float>(0.0);
                                auto tmp22 = tmp10 ? tmp20 : tmp21;
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr1[static_cast<long>(x1 + (23L*x0))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(144L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (144L*x2) + (2304L*x1) + (9216L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>(x1 + (4L*x3) + (4L*x3_inner) + (576L*x2) + (9216L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (16L*x1) + (64L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.25);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (144L*x2) + (2304L*x1) + (9216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_unfold_backward_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       int* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2457600L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(640L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(12L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(12L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x4 + (12L*x3) + (144L*x1) + (576L*x2)), 576L)) % static_cast<long>(80L));
                                auto tmp1 = static_cast<long>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<long>(16);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = [&]
                                {
                                    auto tmp6 = in_ptr0[static_cast<long>(x4 + (12L*x3) + (144L*(static_cast<long>(c10::div_floor_integer((x4 + (12L*x3) + (144L*x1) + (576L*x2)), 576L)) % static_cast<long>(80L))) + (2304L*x1) + (2304L*(c10::div_floor_integer((x4 + (12L*x3)), 144L))) + (9216L*(c10::div_floor_integer((x4 + (12L*x3) + (144L*x1) + (576L*x2)), 46080L))) + (73728L*x0))];
                                    return tmp6;
                                }
                                ;
                                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                                auto tmp8 = tmp0 >= tmp3;
                                auto tmp9 = static_cast<long>(80);
                                auto tmp10 = tmp0 < tmp9;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = in_ptr1[static_cast<long>((-16L) + (64L*x4) + (768L*x3) + (9216L*x1) + (9216L*(c10::div_floor_integer((x4 + (12L*x3)), 144L))) + (36864L*(c10::div_floor_integer((x4 + (12L*x3) + (144L*x1) + (576L*x2)), 46080L))) + (294912L*x0) + (static_cast<long>(c10::div_floor_integer((x4 + (12L*x3) + (144L*x1) + (576L*x2)), 576L)) % static_cast<long>(80L)))];
                                    return tmp12;
                                }
                                ;
                                auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                                auto tmp14 = tmp4 ? tmp7 : tmp13;
                                out_ptr1[static_cast<long>(x3 + (12L*x4) + (144L*x1) + (576L*x2) + (368640L*x0))] = tmp14;
                            }
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                        out_ptr2[static_cast<long>(x1 + (12L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_unfold_backward_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048000L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(10240L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (12L*x2) + (240L*x0)), static_cast<long>(12L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (20L*x1) + (20L*x1_inner) + (240L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(16L); x2<static_cast<long>(20L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (12L*x2) + (240L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (20L*x1) + (20L*x1_inner) + (240L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(20L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (12L*x2) + (240L*x0))];
                        out_ptr1[static_cast<long>(x2 + (20L*x1) + (240L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_constant_pad_nd_convolution_backward_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(5120L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(2L + x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(20);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = c10::convert<long>(2L + x2);
                        auto tmp6 = tmp5 >= tmp1;
                        auto tmp7 = tmp5 < tmp3;
                        auto tmp8 = tmp2 & tmp4;
                        auto tmp9 = tmp8 & tmp6;
                        auto tmp10 = tmp9 & tmp7;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr0[static_cast<long>(42L + x2 + (20L*x1) + (400L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        out_ptr0[static_cast<long>(x2 + (16L*x1) + (256L*x0))] = tmp13;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((16L*(static_cast<long>(c10::div_floor_integer(((4L*(static_cast<long>(x1) % static_cast<long>(4L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (64L*x2) + (64L*x2_inner) + (512L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1)), 32L))) + (1024L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 1024L))) + (8192L*x0) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 64L)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((16L*(static_cast<long>((x2 + x2_inner)) % static_cast<long>(4L))) + (64L*(static_cast<long>(c10::div_floor_integer(((4L*(static_cast<long>(x1) % static_cast<long>(4L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (256L*(c10::div_floor_integer((x2 + x2_inner), 4L))) + (512L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1)), 32L))) + (1024L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 1024L))) + (8192L*x0) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 64L)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((16L*(static_cast<long>((x2 + x2_inner)) % static_cast<long>(4L))) + (64L*(static_cast<long>(x1) % static_cast<long>(4L))) + (256L*(c10::div_floor_integer((x2 + x2_inner), 4L))) + (512L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1)), 32L))) + (1024L*(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 1024L))) + (8192L*x0) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 64L)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp4.store(out_ptr0 + static_cast<long>(x2 + (8L*x1) + (64L*x3) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_batch_norm_backward_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(0.00048828125);
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_batch_norm_backward_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(0.00048828125);
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
                    tmp22.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_mul_native_batch_norm_backward_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>((256L*(static_cast<long>(x1) % static_cast<long>(8L))) + (2048L*(static_cast<long>(x0) % static_cast<long>(2L))) + (4096L*(c10::div_floor_integer(x1, 8L))) + (32768L*(c10::div_floor_integer(x0, 2L))) + (65536L*(c10::div_floor_integer((x3 + x3_inner + (32L*x2)), 256L))) + (static_cast<long>((x3 + x3_inner + (32L*x2))) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>((256L*(static_cast<long>(x1) % static_cast<long>(8L))) + (2048L*(static_cast<long>(x0) % static_cast<long>(2L))) + (4096L*(c10::div_floor_integer(x1, 8L))) + (32768L*(c10::div_floor_integer(x0, 2L))) + (65536L*(c10::div_floor_integer((x3 + x3_inner + (32L*x2)), 256L))) + (static_cast<long>((x3 + x3_inner + (32L*x2))) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>((256L*(static_cast<long>(x1) % static_cast<long>(8L))) + (2048L*(static_cast<long>(x0) % static_cast<long>(2L))) + (4096L*(c10::div_floor_integer(x1, 8L))) + (32768L*(c10::div_floor_integer(x0, 2L))) + (65536L*(c10::div_floor_integer((x3 + x3_inner + (32L*x2)), 256L))) + (static_cast<long>((x3 + x3_inner + (32L*x2))) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = out_ptr1[static_cast<long>(static_cast<long>((x3 + x3_inner + (32L*x2))) % static_cast<long>(256L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp8 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr3[static_cast<long>(static_cast<long>((x3 + x3_inner + (32L*x2))) % static_cast<long>(256L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp13 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = out_ptr0[static_cast<long>(static_cast<long>((x3 + x3_inner + (32L*x2))) % static_cast<long>(256L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp16 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr4[static_cast<long>(static_cast<long>((x3 + x3_inner + (32L*x2))) % static_cast<long>(256L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp5 = static_cast<float>(0.00048828125);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = tmp8 * tmp8;
                            auto tmp10 = tmp7 * tmp9;
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp12 = tmp2 - tmp11;
                            auto tmp14 = tmp13 * tmp6;
                            auto tmp15 = tmp12 - tmp14;
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp18 = tmp15 * tmp17;
                            tmp18.store(out_ptr3 + static_cast<long>(x3 + (32L*x1) + (2048L*x0) + (8192L*x2)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_view_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(144L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (144L*x2) + (9216L*x1) + (36864L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (4L*x3) + (4L*x3_inner) + (576L*x2) + (36864L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (64L*x1) + (256L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(12L); x3+=static_cast<long>(1L))
                        {
                            {
                                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                                float tmp_acc0 = 0;
                                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                                for(long x4=static_cast<long>(0L); x4<static_cast<long>(8L); x4+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x4 + (12L*x3) + (144L*x2) + (1152L*x1) + (9216L*x0)));
                                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x4_inner = 0; x4_inner < 8; x4_inner++) tmpbuf[x4_inner] = in_ptr1[static_cast<long>((4L*x4) + (4L*x4_inner) + (48L*x3) + (576L*x2) + (4608L*x1) + (36864L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                    auto tmp3 = out_ptr0[static_cast<long>(x2 + (8L*x1) + (64L*x0))];
                                    auto tmp2 = tmp0 * tmp1;
                                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                    auto tmp5 = tmp1 * tmp4;
                                    auto tmp6 = tmp2 - tmp5;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                }
                                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                                for(long x4=static_cast<long>(8L); x4<static_cast<long>(12L); x4+=static_cast<long>(1L))
                                {
                                    auto tmp0 = in_ptr0[static_cast<long>(x4 + (12L*x3) + (144L*x2) + (1152L*x1) + (9216L*x0))];
                                    auto tmp1 = in_ptr1[static_cast<long>((4L*x4) + (48L*x3) + (576L*x2) + (4608L*x1) + (36864L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L)))];
                                    auto tmp3 = out_ptr0[static_cast<long>(x2 + (8L*x1) + (64L*x0))];
                                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                    tmp_acc0 = tmp_acc0 + tmp5;
                                }
                                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                                out_ptr1[static_cast<long>(x3 + (12L*x1) + (96L*x2) + (768L*x0))] = static_cast<float>(tmp_acc0);
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(23L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(24);
                    auto tmp2 = tmp0 < tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<long>(x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L))));
                        auto tmp5 = static_cast<long>(207);
                        auto tmp6 = tmp4 < tmp5;
                        auto tmp7 = [&]
                        {
                            auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L)))), 23L)) % static_cast<long>(9L));
                            auto tmp9 = static_cast<long>(8);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L))))) % static_cast<long>(23L));
                                auto tmp13 = static_cast<long>(11);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = [&]
                                {
                                    auto tmp16 = out_ptr1[static_cast<long>((-11L) + (12L*(static_cast<long>(c10::div_floor_integer((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L)))), 23L)) % static_cast<long>(9L))) + (96L*(c10::div_floor_integer(x0, 8L))) + (static_cast<long>((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L))))) % static_cast<long>(23L)))];
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp14 ? tmp17 : tmp18;
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp21 = static_cast<float>(0.0);
                            auto tmp22 = tmp10 ? tmp20 : tmp21;
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                    out_ptr2[static_cast<long>(x1 + (23L*x0))] = tmp24;
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(12L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (12L*x3) + (144L*x1) + (9216L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((4L*x2) + (4L*x2_inner) + (48L*x3) + (576L*x1) + (36864L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = in_ptr2[static_cast<long>(x1 + (64L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x2 + (12L*x1) + (768L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(12L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (12L*x3) + (144L*x1) + (9216L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>((4L*x2) + (48L*x3) + (576L*x1) + (36864L*(c10::div_floor_integer(x0, 4L))) + (static_cast<long>(x0) % static_cast<long>(4L)))];
                                auto tmp3 = in_ptr2[static_cast<long>(x1 + (64L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr0[static_cast<long>(x2 + (12L*x1) + (768L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(23L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(24);
                    auto tmp2 = tmp0 < tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<long>(x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L))));
                        auto tmp5 = static_cast<long>(207);
                        auto tmp6 = tmp4 < tmp5;
                        auto tmp7 = [&]
                        {
                            auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L)))), 23L)) % static_cast<long>(9L));
                            auto tmp9 = static_cast<long>(8);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L))))) % static_cast<long>(23L));
                                auto tmp13 = static_cast<long>(11);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = [&]
                                {
                                    auto tmp16 = out_ptr0[static_cast<long>((-11L) + (12L*(static_cast<long>(c10::div_floor_integer((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L)))), 23L)) % static_cast<long>(9L))) + (96L*(c10::div_floor_integer(x0, 8L))) + (static_cast<long>((x1 + (24L*(static_cast<long>(x0) % static_cast<long>(8L))))) % static_cast<long>(23L)))];
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp14 ? tmp17 : tmp18;
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp21 = static_cast<float>(0.0);
                            auto tmp22 = tmp10 ? tmp20 : tmp21;
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                    out_ptr1[static_cast<long>(x1 + (23L*x0))] = tmp24;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(144L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (144L*x2) + (9216L*x1) + (36864L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>(x1 + (4L*x3) + (4L*x3_inner) + (576L*x2) + (36864L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (256L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.25);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (144L*x2) + (9216L*x1) + (36864L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_unfold_backward_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1474560L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(12L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(12L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x4 + (12L*x3) + (144L*x1) + (576L*x2)), 576L)) % static_cast<long>(48L));
                                auto tmp1 = static_cast<long>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<long>(16);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = [&]
                                {
                                    auto tmp6 = in_ptr0[static_cast<long>(x4 + (12L*x3) + (144L*(static_cast<long>(c10::div_floor_integer((x4 + (12L*x3) + (144L*x1) + (576L*x2)), 576L)) % static_cast<long>(48L))) + (2304L*x1) + (2304L*(c10::div_floor_integer((x4 + (12L*x3)), 144L))) + (9216L*(c10::div_floor_integer((x4 + (12L*x3) + (144L*x1) + (576L*x2)), 27648L))) + (73728L*x0))];
                                    return tmp6;
                                }
                                ;
                                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                                auto tmp8 = tmp0 >= tmp3;
                                auto tmp9 = static_cast<long>(48);
                                auto tmp10 = tmp0 < tmp9;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = in_ptr1[static_cast<long>((-16L) + (32L*x4) + (384L*x3) + (4608L*x1) + (4608L*(c10::div_floor_integer((x4 + (12L*x3)), 144L))) + (18432L*(c10::div_floor_integer((x4 + (12L*x3) + (144L*x1) + (576L*x2)), 27648L))) + (147456L*x0) + (static_cast<long>(c10::div_floor_integer((x4 + (12L*x3) + (144L*x1) + (576L*x2)), 576L)) % static_cast<long>(48L)))];
                                    return tmp12;
                                }
                                ;
                                auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                                auto tmp14 = tmp4 ? tmp7 : tmp13;
                                out_ptr1[static_cast<long>(x3 + (12L*x4) + (144L*x1) + (576L*x2) + (221184L*x0))] = tmp14;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_unfold_backward_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1228800L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (12L*x2) + (240L*x0)), static_cast<long>(12L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (20L*x1) + (20L*x1_inner) + (240L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(16L); x2<static_cast<long>(20L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (12L*x2) + (240L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (20L*x1) + (20L*x1_inner) + (240L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(20L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (12L*x2) + (240L*x0))];
                        out_ptr1[static_cast<long>(x2 + (20L*x1) + (240L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_constant_pad_nd_convolution_backward_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(2L + x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(20);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = c10::convert<long>(2L + x2);
                        auto tmp6 = tmp5 >= tmp1;
                        auto tmp7 = tmp5 < tmp3;
                        auto tmp8 = tmp2 & tmp4;
                        auto tmp9 = tmp8 & tmp6;
                        auto tmp10 = tmp9 & tmp7;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr0[static_cast<long>(42L + x2 + (20L*x1) + (400L*x0))];
                            return tmp12;
                        }
                        ;
                        auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        out_ptr0[static_cast<long>(x2 + (16L*x1) + (256L*x0))] = tmp13;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_29 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((16L*(static_cast<long>(c10::div_floor_integer(((8L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(8L))), 8L)) % static_cast<long>(8L))) + (128L*x2) + (128L*x2_inner) + (2048L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1)), 128L))) + (4096L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 4096L))) + (32768L*x0) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((16L*(static_cast<long>((x2 + x2_inner)) % static_cast<long>(8L))) + (128L*(static_cast<long>(c10::div_floor_integer(((8L*(static_cast<long>(x1) % static_cast<long>(8L))) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(8L))), 8L)) % static_cast<long>(8L))) + (1024L*(c10::div_floor_integer((x2 + x2_inner), 8L))) + (2048L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1)), 128L))) + (4096L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 4096L))) + (32768L*x0) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((16L*(static_cast<long>((x2 + x2_inner)) % static_cast<long>(8L))) + (128L*(static_cast<long>(x1) % static_cast<long>(8L))) + (1024L*(c10::div_floor_integer((x2 + x2_inner), 8L))) + (2048L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1)), 128L))) + (4096L*(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 4096L))) + (32768L*x0) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp4.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (256L*x3) + (32768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_batch_norm_backward_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(0.00048828125);
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_31 = async_compile.cpp('''
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
                       const float* in_ptr11,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp0 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp8;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00048828125);
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
                    auto tmp21 = tmp19 - tmp20;
                    auto tmp23 = tmp22 * tmp6;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp21 * tmp26;
                    auto tmp28 = tmp0 - tmp27;
                    auto tmp29 = tmp28 - tmp14;
                    auto tmp31 = tmp24 * tmp30;
                    auto tmp32 = tmp29 * tmp31;
                    tmp18.store(out_ptr4 + static_cast<long>(x1 + (1024L*x0)));
                    tmp32.store(out_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (65536L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x2) + (65536L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x2) + (65536L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x2) + (65536L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x2) + (65536L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(256.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(256.0);
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
                        auto tmp22 = static_cast<float>(0.00048828125);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0001220703125);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_batch_norm_backward_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
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
                    tmp22.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (131072L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_38 = async_compile.cpp('''
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x2) + (131072L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x2) + (131072L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x2) + (131072L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(1024.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(1024.0);
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
                        auto tmp22 = static_cast<float>(0.0001220703125);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_39 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0001220703125);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_40 = async_compile.cpp('''
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
                       const float* in_ptr11,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp0 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp8;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.0001220703125);
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
                    auto tmp21 = tmp19 - tmp20;
                    auto tmp23 = tmp22 * tmp6;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp21 * tmp26;
                    auto tmp28 = tmp0 - tmp27;
                    auto tmp29 = tmp28 - tmp14;
                    auto tmp31 = tmp24 * tmp30;
                    auto tmp32 = tmp29 * tmp31;
                    tmp18.store(out_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    tmp32.store(out_ptr5 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (131072L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x2) + (131072L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x2) + (131072L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x2) + (131072L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x2) + (131072L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(1024.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(1024.0);
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
                        auto tmp22 = static_cast<float>(0.0001220703125);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(3.0517578125e-05);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_batch_norm_backward_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(3.0517578125e-05);
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
                    tmp22.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(4096L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (262144L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (262144L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(4096L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x2) + (262144L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x2) + (262144L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (64L*x2) + (262144L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(4096.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (64L*x1) + (262144L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x1) + (262144L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (64L*x1) + (262144L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(4096.0);
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
                        auto tmp22 = static_cast<float>(3.0517578125e-05);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (64L*x1) + (262144L*x0)));
                    }
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(3.0517578125e-05);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_49 = async_compile.cpp('''
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
                       const float* in_ptr11,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp0 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp8;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(3.0517578125e-05);
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
                    auto tmp21 = tmp19 - tmp20;
                    auto tmp23 = tmp22 * tmp6;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp21 * tmp26;
                    auto tmp28 = tmp0 - tmp27;
                    auto tmp29 = tmp28 - tmp14;
                    auto tmp31 = tmp24 * tmp30;
                    auto tmp32 = tmp29 * tmp31;
                    tmp18.store(out_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    tmp32.store(out_ptr5 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(4096L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (262144L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (262144L*x0)));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = tmp0 * tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(4096L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x2) + (262144L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x2) + (262144L*x1)));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (64L*x2) + (262144L*x1)));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = static_cast<float>(4096.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (64L*x1) + (262144L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x1) + (262144L*x0)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (64L*x1) + (262144L*x0)));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = static_cast<float>(4096.0);
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
                        auto tmp22 = static_cast<float>(3.0517578125e-05);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 * tmp23;
                        auto tmp26 = tmp25 * tmp25;
                        auto tmp27 = tmp24 * tmp26;
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp29 = tmp17 - tmp28;
                        auto tmp31 = tmp30 * tmp23;
                        auto tmp32 = tmp29 - tmp31;
                        tmp32.store(in_out_ptr0 + static_cast<long>(x2 + (64L*x1) + (262144L*x0)));
                    }
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(3.0517578125e-05);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(131072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(7.62939453125e-06);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_56 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(131072L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(7.62939453125e-06);
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


cpp_fused_convolution_backward_mul_native_batch_norm_backward_57 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(131072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (24L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(7.62939453125e-06);
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_47, primals_49, primals_51, primals_55, primals_57, primals_59, primals_61, primals_65, primals_67, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_203, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, getitem_6, getitem_7, convolution_3, squeeze_10, mul_31, convolution_4, squeeze_13, add_24, view, convolution_5, mul_40, convolution_6, squeeze_16, convolution_7, squeeze_19, mul_55, convolution_8, squeeze_22, mul_63, convolution_9, squeeze_25, add_45, view_2, convolution_10, mul_72, convolution_11, squeeze_28, mul_80, convolution_12, squeeze_31, mul_88, convolution_13, squeeze_34, add_61, view_4, convolution_14, mul_97, convolution_15, squeeze_37, convolution_16, squeeze_40, mul_112, convolution_17, squeeze_43, mul_120, convolution_18, squeeze_46, add_82, view_6, convolution_19, mul_129, convolution_20, squeeze_49, mul_137, convolution_21, squeeze_52, mul_145, convolution_22, squeeze_55, add_98, view_8, convolution_23, mul_154, convolution_24, squeeze_58, convolution_25, squeeze_61, mul_169, convolution_26, squeeze_64, mul_177, view_17, view_23, squeeze_67, mul_186, convolution_29, squeeze_70, mul_194, convolution_30, squeeze_73, mul_202, view_42, view_48, squeeze_76, mul_211, convolution_33, squeeze_79, convolution_34, squeeze_82, mul_226, convolution_35, squeeze_85, mul_234, view_67, view_73, squeeze_88, mul_243, convolution_38, squeeze_91, clone_51, permute_34, mul_253, unsqueeze_126, mul_265, sub_40, permute_42, permute_43, alias_8, permute_47, permute_53, permute_55, permute_56, mul_280, unsqueeze_150, mul_292, unsqueeze_162, unsqueeze_174, mul_313, sub_60, permute_68, permute_69, alias_9, permute_73, permute_79, permute_81, permute_82, mul_328, unsqueeze_198, mul_340, unsqueeze_210, mul_352, sub_76, permute_94, permute_95, alias_10, permute_99, permute_105, permute_107, permute_108, mul_367, unsqueeze_234, mul_379, unsqueeze_246, unsqueeze_258, unsqueeze_272, mul_416, unsqueeze_284, mul_428, unsqueeze_296, unsqueeze_310, mul_456, unsqueeze_322, mul_468, unsqueeze_334, unsqueeze_346, unsqueeze_360, mul_505, unsqueeze_372, mul_517, unsqueeze_384, unsqueeze_398, mul_545, unsqueeze_410, mul_557, unsqueeze_422, unsqueeze_434, unsqueeze_448, mul_594, unsqueeze_460, mul_606, unsqueeze_472, mul_618, unsqueeze_484, mul_630, unsqueeze_496, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (24, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_13, (256, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_37, (256, ), (1, ))
    assert_size_stride(primals_39, (1024, ), (1, ))
    assert_size_stride(primals_41, (1024, ), (1, ))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_49, (1024, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_55, (512, ), (1, ))
    assert_size_stride(primals_57, (2048, ), (1, ))
    assert_size_stride(primals_59, (2048, ), (1, ))
    assert_size_stride(primals_61, (512, ), (1, ))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_67, (2048, ), (1, ))
    assert_size_stride(primals_69, (24, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_70, (32, 24, 3, 3), (216, 1, 72, 24))
    assert_size_stride(primals_71, (64, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_72, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_73, (64, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_74, (1, 1, 3), (3, 3, 1))
    assert_size_stride(primals_75, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_76, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_77, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_78, (64, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_79, (1, 1, 3), (3, 3, 1))
    assert_size_stride(primals_80, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_81, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_82, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_83, (1, 1, 5), (5, 5, 1))
    assert_size_stride(primals_84, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_85, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_86, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_87, (128, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_88, (1, 1, 5), (5, 5, 1))
    assert_size_stride(primals_89, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_90, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_91, (256, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_92, (1, 1, 5), (5, 5, 1))
    assert_size_stride(primals_93, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_94, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_95, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_96, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_97, (384, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_98, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_99, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_100, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_101, (640, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_102, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_103, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_104, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_105, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_106, (640, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_107, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_203, (8, 3, 256, 256), (196608, 1, 768, 3))
    assert_size_stride(convolution, (8, 24, 128, 128), (393216, 1, 3072, 24))
    assert_size_stride(squeeze_1, (24, ), (1, ))
    assert_size_stride(mul_7, (8, 24, 128, 128), (393216, 1, 3072, 24))
    assert_size_stride(convolution_1, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(mul_15, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(convolution_2, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(mul_23, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(getitem_6, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(getitem_7, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_3, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_10, (64, ), (1, ))
    assert_size_stride(mul_31, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_4, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_13, (64, ), (1, ))
    assert_size_stride(add_24, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(view, (8, 1, 64), (64, 64, 1))
    assert_size_stride(convolution_5, (8, 1, 64), (64, 64, 1))
    assert_size_stride(mul_40, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_6, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(squeeze_16, (256, ), (1, ))
    assert_size_stride(convolution_7, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(squeeze_19, (256, ), (1, ))
    assert_size_stride(mul_55, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(convolution_8, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_22, (64, ), (1, ))
    assert_size_stride(mul_63, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_9, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_25, (64, ), (1, ))
    assert_size_stride(add_45, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(view_2, (8, 1, 64), (64, 64, 1))
    assert_size_stride(convolution_10, (8, 1, 64), (64, 64, 1))
    assert_size_stride(mul_72, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_11, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(squeeze_28, (256, ), (1, ))
    assert_size_stride(mul_80, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(convolution_12, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(squeeze_31, (128, ), (1, ))
    assert_size_stride(mul_88, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(convolution_13, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_34, (128, ), (1, ))
    assert_size_stride(add_61, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(view_4, (8, 1, 128), (128, 128, 1))
    assert_size_stride(convolution_14, (8, 1, 128), (128, 128, 1))
    assert_size_stride(mul_97, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_15, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(squeeze_37, (512, ), (1, ))
    assert_size_stride(convolution_16, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(squeeze_40, (512, ), (1, ))
    assert_size_stride(mul_112, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(convolution_17, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_43, (128, ), (1, ))
    assert_size_stride(mul_120, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_18, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_46, (128, ), (1, ))
    assert_size_stride(add_82, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(view_6, (8, 1, 128), (128, 128, 1))
    assert_size_stride(convolution_19, (8, 1, 128), (128, 128, 1))
    assert_size_stride(mul_129, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_20, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(squeeze_49, (512, ), (1, ))
    assert_size_stride(mul_137, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(convolution_21, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(squeeze_52, (256, ), (1, ))
    assert_size_stride(mul_145, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(convolution_22, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_55, (256, ), (1, ))
    assert_size_stride(add_98, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(view_8, (8, 1, 256), (256, 256, 1))
    assert_size_stride(convolution_23, (8, 1, 256), (256, 256, 1))
    assert_size_stride(mul_154, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_24, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(squeeze_58, (1024, ), (1, ))
    assert_size_stride(convolution_25, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(squeeze_61, (1024, ), (1, ))
    assert_size_stride(mul_169, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(convolution_26, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_64, (256, ), (1, ))
    assert_size_stride(mul_177, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(view_17, (16384, 16), (16, 1))
    assert_size_stride(view_23, (16384, 16), (16, 1))
    assert_size_stride(squeeze_67, (256, ), (1, ))
    assert_size_stride(mul_186, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_29, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(squeeze_70, (1024, ), (1, ))
    assert_size_stride(mul_194, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(convolution_30, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(squeeze_73, (512, ), (1, ))
    assert_size_stride(mul_202, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(view_42, (4096, 16), (16, 1))
    assert_size_stride(view_48, (4096, 16), (16, 1))
    assert_size_stride(squeeze_76, (512, ), (1, ))
    assert_size_stride(mul_211, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_33, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(squeeze_79, (2048, ), (1, ))
    assert_size_stride(convolution_34, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(squeeze_82, (2048, ), (1, ))
    assert_size_stride(mul_226, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(convolution_35, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_85, (512, ), (1, ))
    assert_size_stride(mul_234, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(view_67, (4096, 16), (16, 1))
    assert_size_stride(view_73, (4096, 16), (16, 1))
    assert_size_stride(squeeze_88, (512, ), (1, ))
    assert_size_stride(mul_243, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_38, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(squeeze_91, (2048, ), (1, ))
    assert_size_stride(clone_51, (8, 2048), (2048, 1))
    assert_size_stride(permute_34, (1000, 2048), (2048, 1))
    assert_size_stride(mul_253, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(unsqueeze_126, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(mul_265, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(sub_40, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(permute_42, (64, 144, 64), (9216, 1, 144))
    assert_size_stride(permute_43, (64, 64, 144), (64, 1, 4096))
    assert_size_stride(alias_8, (64, 1, 64, 144), (9216, 9216, 144, 1))
    assert_size_stride(permute_47, (23, 16), (16, 1))
    assert_size_stride(permute_53, (23, 16), (16, 1))
    assert_size_stride(permute_55, (64, 16, 64), (1024, 64, 1))
    assert_size_stride(permute_56, (64, 144, 16), (16, 1024, 1))
    assert_size_stride(mul_280, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_150, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_292, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(unsqueeze_162, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_174, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(mul_313, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(sub_60, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(permute_68, (256, 144, 16), (2304, 1, 144))
    assert_size_stride(permute_69, (256, 64, 144), (9216, 1, 64))
    assert_size_stride(alias_9, (64, 4, 16, 144), (9216, 1, 576, 4))
    assert_size_stride(permute_73, (23, 16), (16, 1))
    assert_size_stride(permute_79, (23, 16), (16, 1))
    assert_size_stride(permute_81, (256, 16, 16), (256, 1, 16))
    assert_size_stride(permute_82, (256, 144, 16), (2304, 1, 144))
    assert_size_stride(mul_328, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(unsqueeze_198, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_340, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(unsqueeze_210, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(mul_352, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(sub_76, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(permute_94, (256, 144, 64), (9216, 1, 144))
    assert_size_stride(permute_95, (256, 32, 144), (4608, 1, 32))
    assert_size_stride(alias_10, (64, 4, 64, 144), (36864, 1, 576, 4))
    assert_size_stride(permute_99, (23, 16), (16, 1))
    assert_size_stride(permute_105, (23, 16), (16, 1))
    assert_size_stride(permute_107, (256, 16, 64), (1024, 1, 16))
    assert_size_stride(permute_108, (256, 144, 16), (2304, 1, 144))
    assert_size_stride(mul_367, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_234, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_379, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(unsqueeze_246, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_272, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_416, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(unsqueeze_284, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_428, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(unsqueeze_296, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_310, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_456, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_322, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_468, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(unsqueeze_334, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_346, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_360, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_505, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(unsqueeze_372, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_517, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(unsqueeze_384, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_398, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_545, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(unsqueeze_410, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_557, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(unsqueeze_422, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_434, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_448, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_594, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(unsqueeze_460, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_606, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    assert_size_stride(unsqueeze_472, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_618, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(unsqueeze_484, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(mul_630, (8, 24, 128, 128), (393216, 1, 3072, 24))
    assert_size_stride(unsqueeze_496, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_34, out=buf0)
    del permute_34
    buf1 = empty((1000, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_51, out=buf1)
    del clone_51
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf4 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf5 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf6 = empty((8, 2048, 8, 8), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_mul_native_batch_norm_backward_sum_0(c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(mul_253.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(unsqueeze_126.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del convolution_38
    del primals_67
    del squeeze_91
    del tangents_1
    del unsqueeze_126
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
    buf7 = aten.convolution_backward(buf6, mul_243, primals_107, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_243
    del primals_107
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf10 = empty((512, ), device='cpu', dtype=torch.float32)
    buf11 = empty((512, ), device='cpu', dtype=torch.float32)
    buf12 = empty((512, ), device='cpu', dtype=torch.float32)
    buf13 = empty((8, 512, 8, 8), device='cpu', dtype=torch.float32)
    cpp_fused_mul_native_batch_norm_backward_1(c_void_p(buf8.data_ptr()), c_void_p(mul_265.data_ptr()), c_void_p(sub_40.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    del buf8
    del mul_265
    del primals_65
    del squeeze_88
    del sub_40
    buf14 = empty((64, 144, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_42, reinterpret_tensor(buf13, (64, 64, 64), (4096, 1, 64), 0), out=buf14)
    del permute_42
    buf15 = empty((64, 64, 144), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf13, (64, 64, 64), (4096, 1, 64), 0), permute_43, out=buf15)
    del permute_43
    buf16 = empty_strided((64, 1, 64, 1), (64, 4096, 1, 4096), device='cpu', dtype=torch.float32)
    buf17 = empty((64, 8, 1, 8, 12), device='cpu', dtype=torch.float32)
    buf18 = empty((4096, 23), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_sum_view_2(c_void_p(buf15.data_ptr()), c_void_p(alias_8.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()))
    buf19 = empty((23, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf18, (23, 4096), (1, 23), 0), view_73, out=buf19)
    del view_73
    buf20 = empty((4096, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf18, permute_47, out=buf20)
    del permute_47
    buf21 = buf17; del buf17  # reuse
    buf22 = buf18; del buf18  # reuse
    cpp_fused_sum_view_3(c_void_p(buf15.data_ptr()), c_void_p(alias_8.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()))
    buf23 = empty((23, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf22, (23, 4096), (1, 23), 0), view_67, out=buf23)
    del view_67
    buf24 = empty((4096, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf22, permute_53, out=buf24)
    del permute_53
    buf25 = reinterpret_tensor(buf15, (64, 1, 64, 144), (9216, 9216, 144, 1), 0); del buf15  # reuse
    cpp_fused__softmax_backward_data_mul_4(c_void_p(buf25.data_ptr()), c_void_p(alias_8.data_ptr()), c_void_p(buf16.data_ptr()))
    del alias_8
    buf26 = empty((64, 16, 144), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_55, reinterpret_tensor(buf25, (64, 64, 144), (9216, 144, 1), 0), out=buf26)
    del permute_55
    buf27 = empty((64, 64, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf25, (64, 64, 144), (9216, 144, 1), 0), permute_56, out=buf27)
    del permute_56
    buf28 = empty((8, 640, 1, 12, 12), device='cpu', dtype=torch.float32)
    buf29 = empty((64, 80, 1, 144), device='cpu', dtype=torch.float32)
    buf30 = reinterpret_tensor(buf29, (8, 640, 1, 12, 12), (92160, 144, 737280, 1, 12), 0); del buf29  # reuse
    buf31 = empty((12, ), device='cpu', dtype=torch.int32)
    cpp_fused_clone_unfold_backward_5(c_void_p(buf30.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf31.data_ptr()))
    del buf26
    aten.index_put_(buf28, [None, None, None, reinterpret_tensor(buf31, (12, ), (1, ), 0)], buf30, True)
    buf34 = reinterpret_tensor(buf30, (8, 640, 12, 12), (92160, 144, 12, 1), 0); del buf30  # reuse
    cpp_fused_unfold_backward_6(c_void_p(buf34.data_ptr()))
    aten.index_put_(buf34, [None, None, reinterpret_tensor(buf31, (12, ), (1, ), 0)], reinterpret_tensor(buf28, (8, 640, 12, 12), (92160, 144, 1, 12), 0), True)
    del buf28
    del buf31
    buf37 = empty((8, 640, 8, 8), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_convolution_backward_7(c_void_p(buf34.data_ptr()), c_void_p(buf37.data_ptr()))
    del buf34
    # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward]
    buf38 = aten.convolution_backward(buf37, mul_234, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf37
    del primals_106
    buf39 = buf38[0]
    buf40 = buf38[1]
    del buf38
    buf41 = empty((8, 128, 8, 8), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_8(c_void_p(buf20.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf41.data_ptr()))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf42 = aten.convolution_backward(buf41, mul_234, primals_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_234
    del primals_105
    buf43 = buf42[0]
    buf44 = buf42[1]
    del buf42
    buf45 = buf11; del buf11  # reuse
    buf46 = empty((512, ), device='cpu', dtype=torch.float32)
    buf47 = buf39; del buf39  # reuse
    buf48 = buf46; del buf46  # reuse
    cpp_fused_add_mul_native_batch_norm_backward_9(c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(mul_280.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_150.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf45.data_ptr()))
    del convolution_35
    del mul_280
    del primals_61
    del squeeze_85
    del unsqueeze_150
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf49 = aten.convolution_backward(buf47, mul_226, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_226
    del primals_104
    buf50 = buf49[0]
    buf51 = buf49[1]
    del buf49
    buf52 = buf4; del buf4  # reuse
    buf53 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf60 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf54 = reinterpret_tensor(buf6, (8, 2048, 8, 8), (131072, 1, 16384, 2048), 0); del buf6  # reuse
    buf61 = empty_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    buf55 = buf53; del buf53  # reuse
    buf56 = empty((8, 2048, 8, 8), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_10(c_void_p(buf55.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(mul_253.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(mul_292.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_162.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_174.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf56.data_ptr()))
    del buf50
    del buf54
    del convolution_33
    del convolution_34
    del mul_253
    del mul_292
    del primals_59
    del squeeze_82
    del unsqueeze_162
    del unsqueeze_174
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf57 = aten.convolution_backward(buf56, mul_194, primals_103, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_103
    buf58 = buf57[0]
    buf59 = buf57[1]
    del buf57
    buf62 = buf60; del buf60  # reuse
    buf63 = buf56; del buf56  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_11(c_void_p(buf62.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf63.data_ptr()))
    del buf61
    del primals_57
    del squeeze_79
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf64 = aten.convolution_backward(buf63, mul_211, primals_102, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf63
    del mul_211
    del primals_102
    buf65 = buf64[0]
    buf66 = buf64[1]
    del buf64
    buf67 = empty((512, ), device='cpu', dtype=torch.float32)
    buf68 = empty((512, ), device='cpu', dtype=torch.float32)
    buf69 = empty((512, ), device='cpu', dtype=torch.float32)
    buf70 = reinterpret_tensor(buf47, (64, 4, 16, 64), (4096, 1024, 64, 1), 0); del buf47  # reuse
    cpp_fused_clone_mul_native_batch_norm_backward_12(c_void_p(buf65.data_ptr()), c_void_p(mul_313.data_ptr()), c_void_p(sub_60.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    del mul_313
    del primals_55
    del squeeze_76
    del sub_60
    buf71 = empty((256, 144, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_68, reinterpret_tensor(buf70, (256, 16, 64), (1024, 64, 1), 0), out=buf71)
    del permute_68
    buf72 = reinterpret_tensor(buf14, (256, 16, 144), (2304, 144, 1), 0); del buf14  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf70, (256, 16, 64), (1024, 64, 1), 0), permute_69, out=buf72)
    del permute_69
    buf73 = reinterpret_tensor(buf16, (64, 4, 16, 1), (64, 16, 1, 4096), 0); del buf16  # reuse
    buf74 = reinterpret_tensor(buf21, (256, 4, 1, 4, 12), (192, 48, 48, 12, 1), 0); del buf21  # reuse
    buf75 = buf22; del buf22  # reuse
    cpp_fused__softmax_backward_data_sum_view_13(c_void_p(buf72.data_ptr()), c_void_p(alias_9.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()))
    buf76 = empty((23, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf75, (23, 4096), (1, 23), 0), view_48, out=buf76)
    del view_48
    buf77 = reinterpret_tensor(buf41, (4096, 16), (16, 1), 0); del buf41  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf75, permute_73, out=buf77)
    del permute_73
    buf78 = buf74; del buf74  # reuse
    buf79 = buf75; del buf75  # reuse
    cpp_fused_sum_view_14(c_void_p(buf72.data_ptr()), c_void_p(alias_9.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    del buf78
    buf80 = empty((23, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf79, (23, 4096), (1, 23), 0), view_42, out=buf80)
    del view_42
    buf81 = reinterpret_tensor(buf27, (4096, 16), (16, 1), 0); del buf27  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf79, permute_79, out=buf81)
    del buf79
    del permute_79
    buf82 = reinterpret_tensor(buf72, (64, 4, 16, 144), (9216, 2304, 144, 1), 0); del buf72  # reuse
    cpp_fused__softmax_backward_data_mul_15(c_void_p(buf82.data_ptr()), c_void_p(alias_9.data_ptr()), c_void_p(buf73.data_ptr()))
    del alias_9
    del buf73
    buf83 = reinterpret_tensor(buf25, (256, 16, 144), (2304, 144, 1), 0); del buf25  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_81, reinterpret_tensor(buf82, (256, 16, 144), (2304, 144, 1), 0), out=buf83)
    del permute_81
    buf84 = reinterpret_tensor(buf24, (256, 16, 16), (256, 16, 1), 0); del buf24  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf82, (256, 16, 144), (2304, 144, 1), 0), permute_82, out=buf84)
    del buf82
    del permute_82
    buf85 = empty((8, 640, 2, 20, 12), device='cpu', dtype=torch.float32)
    buf86 = empty((8, 640, 2, 2, 12, 12), device='cpu', dtype=torch.float32)
    buf87 = empty((2, 12), device='cpu', dtype=torch.int32)
    cpp_fused_unfold_backward_16(c_void_p(buf83.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()))
    aten.index_put_(buf85, [None, None, None, reinterpret_tensor(buf87, (24, ), (1, ), 0)], reinterpret_tensor(buf86, (8, 640, 2, 24, 12), (368640, 576, 288, 12, 1), 0), True)
    del buf86
    buf90 = empty((8, 640, 20, 20), device='cpu', dtype=torch.float32)
    buf91 = empty((8, 640, 2, 12, 20), device='cpu', dtype=torch.float32)
    cpp_fused_unfold_backward_17(c_void_p(buf85.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()))
    del buf85
    aten.index_put_(buf90, [None, None, reinterpret_tensor(buf87, (24, ), (1, ), 0)], reinterpret_tensor(buf91, (8, 640, 24, 20), (307200, 480, 20, 1), 0), True)
    del buf91
    buf94 = empty((8, 640, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_convolution_backward_18(c_void_p(buf90.data_ptr()), c_void_p(buf94.data_ptr()))
    del buf90
    # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward]
    buf95 = aten.convolution_backward(buf94, mul_202, primals_101, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf94
    del primals_101
    buf96 = buf95[0]
    buf97 = buf95[1]
    del buf95
    buf98 = reinterpret_tensor(buf20, (8, 128, 8, 8), (8192, 64, 8, 1), 0); del buf20  # reuse
    cpp_fused_convolution_backward_19(c_void_p(buf77.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf98.data_ptr()))
    del buf77
    del buf81
    del buf84
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf99 = aten.convolution_backward(buf98, mul_202, primals_100, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf98
    del mul_202
    del primals_100
    buf100 = buf99[0]
    buf101 = buf99[1]
    del buf99
    buf102 = buf68; del buf68  # reuse
    buf103 = empty((512, ), device='cpu', dtype=torch.float32)
    buf104 = buf100; del buf100  # reuse
    buf105 = buf103; del buf103  # reuse
    cpp_fused_add_mul_native_batch_norm_backward_20(c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(mul_328.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_198.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf102.data_ptr()))
    del buf96
    del convolution_30
    del mul_328
    del primals_51
    del squeeze_73
    del unsqueeze_198
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf106 = aten.convolution_backward(buf104, mul_194, primals_99, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf104
    del mul_194
    del primals_99
    buf107 = buf106[0]
    buf108 = buf106[1]
    del buf106
    buf109 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf110 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf111 = empty_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    buf112 = buf110; del buf110  # reuse
    cpp_fused_add_mul_native_batch_norm_backward_21(c_void_p(buf112.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(mul_340.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_210.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf111.data_ptr()))
    del convolution_29
    del primals_49
    del squeeze_70
    del unsqueeze_210
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf113 = aten.convolution_backward(buf111, mul_186, primals_98, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_186
    del primals_98
    buf114 = buf113[0]
    buf115 = buf113[1]
    del buf113
    buf116 = empty((256, ), device='cpu', dtype=torch.float32)
    buf117 = empty((256, ), device='cpu', dtype=torch.float32)
    buf118 = empty((256, ), device='cpu', dtype=torch.float32)
    buf119 = empty((64, 4, 64, 32), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_native_batch_norm_backward_22(c_void_p(buf114.data_ptr()), c_void_p(mul_352.data_ptr()), c_void_p(sub_76.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()))
    del buf114
    del mul_352
    del primals_47
    del squeeze_67
    del sub_76
    buf120 = empty((256, 144, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_94, reinterpret_tensor(buf119, (256, 64, 32), (2048, 32, 1), 0), out=buf120)
    del permute_94
    buf121 = reinterpret_tensor(buf71, (256, 64, 144), (9216, 144, 1), 0); del buf71  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf119, (256, 64, 32), (2048, 32, 1), 0), permute_95, out=buf121)
    del buf119
    del permute_95
    buf122 = reinterpret_tensor(buf0, (64, 4, 64, 1), (256, 64, 1, 16384), 0); del buf0  # reuse
    buf123 = empty((256, 8, 1, 8, 12), device='cpu', dtype=torch.float32)
    buf124 = empty((16384, 23), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_sum_view_23(c_void_p(buf121.data_ptr()), c_void_p(alias_10.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()))
    buf125 = empty((23, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf124, (23, 16384), (1, 23), 0), view_23, out=buf125)
    del view_23
    buf126 = reinterpret_tensor(buf70, (16384, 16), (16, 1), 0); del buf70  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf124, permute_99, out=buf126)
    del permute_99
    buf127 = buf123; del buf123  # reuse
    buf128 = buf124; del buf124  # reuse
    cpp_fused_sum_view_24(c_void_p(buf121.data_ptr()), c_void_p(alias_10.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()))
    del buf127
    buf129 = empty((23, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf128, (23, 16384), (1, 23), 0), view_17, out=buf129)
    del view_17
    buf130 = reinterpret_tensor(buf65, (16384, 16), (16, 1), 0); del buf65  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf128, permute_105, out=buf130)
    del buf128
    del permute_105
    buf131 = reinterpret_tensor(buf121, (64, 4, 64, 144), (36864, 9216, 144, 1), 0); del buf121  # reuse
    cpp_fused__softmax_backward_data_mul_25(c_void_p(buf131.data_ptr()), c_void_p(alias_10.data_ptr()), c_void_p(buf122.data_ptr()))
    del alias_10
    del buf122
    buf132 = buf83; del buf83  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_107, reinterpret_tensor(buf131, (256, 64, 144), (9216, 144, 1), 0), out=buf132)
    del permute_107
    buf133 = reinterpret_tensor(buf43, (256, 64, 16), (1024, 16, 1), 0); del buf43  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf131, (256, 64, 144), (9216, 144, 1), 0), permute_108, out=buf133)
    del buf131
    del permute_108
    buf134 = empty((8, 384, 2, 20, 12), device='cpu', dtype=torch.float32)
    buf135 = empty((8, 384, 2, 2, 12, 12), device='cpu', dtype=torch.float32)
    cpp_fused_unfold_backward_26(c_void_p(buf132.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()))
    del buf120
    del buf132
    aten.index_put_(buf134, [None, None, None, reinterpret_tensor(buf87, (24, ), (1, ), 0)], reinterpret_tensor(buf135, (8, 384, 2, 24, 12), (221184, 576, 288, 12, 1), 0), True)
    del buf135
    buf138 = empty((8, 384, 20, 20), device='cpu', dtype=torch.float32)
    buf139 = empty((8, 384, 2, 12, 20), device='cpu', dtype=torch.float32)
    cpp_fused_unfold_backward_27(c_void_p(buf134.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()))
    del buf134
    aten.index_put_(buf138, [None, None, reinterpret_tensor(buf87, (24, ), (1, ), 0)], reinterpret_tensor(buf139, (8, 384, 24, 20), (184320, 480, 20, 1), 0), True)
    del buf139
    del buf87
    buf142 = empty((8, 384, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_convolution_backward_28(c_void_p(buf138.data_ptr()), c_void_p(buf142.data_ptr()))
    del buf138
    # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.convolution_backward]
    buf143 = aten.convolution_backward(buf142, mul_177, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf142
    del primals_97
    buf144 = buf143[0]
    buf145 = buf143[1]
    del buf143
    buf146 = reinterpret_tensor(buf13, (8, 128, 16, 16), (32768, 256, 16, 1), 0); del buf13  # reuse
    cpp_fused_convolution_backward_29(c_void_p(buf126.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf146.data_ptr()))
    del buf126
    del buf130
    del buf133
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf147 = aten.convolution_backward(buf146, mul_177, primals_96, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf146
    del mul_177
    del primals_96
    buf148 = buf147[0]
    buf149 = buf147[1]
    del buf147
    buf150 = buf117; del buf117  # reuse
    buf151 = empty((256, ), device='cpu', dtype=torch.float32)
    buf152 = buf144; del buf144  # reuse
    buf153 = buf151; del buf151  # reuse
    cpp_fused_add_mul_native_batch_norm_backward_30(c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(mul_367.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_234.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf150.data_ptr()))
    del buf148
    del convolution_26
    del mul_367
    del primals_43
    del squeeze_64
    del unsqueeze_234
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf154 = aten.convolution_backward(buf152, mul_169, primals_95, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf152
    del mul_169
    del primals_95
    buf155 = buf154[0]
    buf156 = buf154[1]
    del buf154
    buf157 = buf107; del buf107  # reuse
    buf158 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf159 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf165 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf160 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf161 = buf111; del buf111  # reuse
    buf167 = empty_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_31(c_void_p(buf157.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(mul_340.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(mul_379.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_246.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_258.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf167.data_ptr()))
    del buf155
    del buf157
    del buf58
    del convolution_24
    del convolution_25
    del mul_340
    del mul_379
    del primals_39
    del primals_41
    del squeeze_61
    del unsqueeze_246
    del unsqueeze_258
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf162 = aten.convolution_backward(buf161, mul_137, primals_94, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf161
    del primals_94
    buf163 = buf162[0]
    buf164 = buf162[1]
    del buf162
    buf166 = buf165; del buf165  # reuse
    cpp_fused_native_batch_norm_backward_32(c_void_p(buf166.data_ptr()), c_void_p(squeeze_58.data_ptr()))
    del squeeze_58
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf168 = aten.convolution_backward(buf167, mul_154, primals_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf167
    del mul_154
    del primals_93
    buf169 = buf168[0]
    buf170 = buf168[1]
    del buf168
    buf171 = empty((8, 256, 1, 1), device='cpu', dtype=torch.float32)
    buf172 = reinterpret_tensor(buf171, (8, 1, 256), (256, 256, 1), 0); del buf171  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_33(c_void_p(buf172.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(add_98.data_ptr()), c_void_p(convolution_23.data_ptr()))
    # Source Nodes: [sigmoid_4], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf173 = aten.convolution_backward(buf172, view_8, primals_92, [0], [1], [2], [1], False, [0], 1, [True, True, False])
    del buf172
    del primals_92
    del view_8
    buf174 = buf173[0]
    buf175 = buf173[1]
    del buf173
    buf176 = empty((256, ), device='cpu', dtype=torch.float32)
    buf177 = empty((256, ), device='cpu', dtype=torch.float32)
    buf178 = buf169; del buf169  # reuse
    buf179 = buf177; del buf177  # reuse
    buf180 = buf178; del buf178  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_34(c_void_p(buf180.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(add_98.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(unsqueeze_272.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf176.data_ptr()))
    del add_98
    del buf174
    del convolution_22
    del convolution_23
    del primals_37
    del squeeze_55
    del unsqueeze_272
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf181 = aten.convolution_backward(buf180, mul_145, primals_91, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False])
    del buf180
    del mul_145
    del primals_91
    buf182 = buf181[0]
    buf183 = buf181[1]
    del buf181
    buf184 = empty((256, ), device='cpu', dtype=torch.float32)
    buf185 = empty((256, ), device='cpu', dtype=torch.float32)
    buf186 = empty((256, ), device='cpu', dtype=torch.float32)
    buf187 = buf182; del buf182  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_35(c_void_p(buf187.data_ptr()), c_void_p(mul_416.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_284.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()))
    del convolution_21
    del mul_416
    del primals_35
    del squeeze_52
    del unsqueeze_284
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf188 = aten.convolution_backward(buf187, mul_137, primals_90, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf187
    del mul_137
    del primals_90
    buf189 = buf188[0]
    buf190 = buf188[1]
    del buf188
    buf191 = empty((512, ), device='cpu', dtype=torch.float32)
    buf192 = empty((512, ), device='cpu', dtype=torch.float32)
    buf193 = empty_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    buf194 = buf192; del buf192  # reuse
    cpp_fused_add_mul_native_batch_norm_backward_36(c_void_p(buf194.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(mul_428.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_296.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf193.data_ptr()))
    del convolution_20
    del primals_33
    del squeeze_49
    del unsqueeze_296
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf195 = aten.convolution_backward(buf193, mul_129, primals_89, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_129
    del primals_89
    buf196 = buf195[0]
    buf197 = buf195[1]
    del buf195
    buf198 = reinterpret_tensor(buf159, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf159  # reuse
    buf199 = reinterpret_tensor(buf198, (8, 1, 128), (128, 128, 1), 0); del buf198  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_37(c_void_p(buf199.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(add_82.data_ptr()), c_void_p(convolution_19.data_ptr()))
    # Source Nodes: [sigmoid_3], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf200 = aten.convolution_backward(buf199, view_6, primals_88, [0], [1], [2], [1], False, [0], 1, [True, True, False])
    del buf199
    del primals_88
    del view_6
    buf201 = buf200[0]
    buf202 = buf200[1]
    del buf200
    buf203 = empty((128, ), device='cpu', dtype=torch.float32)
    buf204 = empty((128, ), device='cpu', dtype=torch.float32)
    buf205 = buf196; del buf196  # reuse
    buf206 = buf204; del buf204  # reuse
    buf207 = buf205; del buf205  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_38(c_void_p(buf207.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(add_82.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_310.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf203.data_ptr()))
    del add_82
    del convolution_18
    del convolution_19
    del primals_31
    del squeeze_46
    del unsqueeze_310
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf208 = aten.convolution_backward(buf207, mul_120, primals_87, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf207
    del mul_120
    del primals_87
    buf209 = buf208[0]
    buf210 = buf208[1]
    del buf208
    buf211 = empty((128, ), device='cpu', dtype=torch.float32)
    buf212 = empty((128, ), device='cpu', dtype=torch.float32)
    buf213 = empty((128, ), device='cpu', dtype=torch.float32)
    buf214 = buf209; del buf209  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_39(c_void_p(buf214.data_ptr()), c_void_p(mul_456.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(unsqueeze_322.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()))
    del convolution_17
    del mul_456
    del primals_29
    del squeeze_43
    del unsqueeze_322
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf215 = aten.convolution_backward(buf214, mul_112, primals_86, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf214
    del mul_112
    del primals_86
    buf216 = buf215[0]
    buf217 = buf215[1]
    del buf215
    buf218 = buf163; del buf163  # reuse
    buf219 = empty((512, ), device='cpu', dtype=torch.float32)
    buf220 = empty((512, ), device='cpu', dtype=torch.float32)
    buf226 = empty((512, ), device='cpu', dtype=torch.float32)
    buf221 = empty((512, ), device='cpu', dtype=torch.float32)
    buf222 = buf193; del buf193  # reuse
    buf228 = empty_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_40(c_void_p(buf218.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(mul_428.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(mul_468.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_334.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_346.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf228.data_ptr()))
    del buf189
    del buf216
    del buf218
    del convolution_15
    del convolution_16
    del mul_428
    del mul_468
    del primals_25
    del primals_27
    del squeeze_40
    del unsqueeze_334
    del unsqueeze_346
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf223 = aten.convolution_backward(buf222, mul_80, primals_85, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf222
    del primals_85
    buf224 = buf223[0]
    buf225 = buf223[1]
    del buf223
    buf227 = buf226; del buf226  # reuse
    cpp_fused_native_batch_norm_backward_41(c_void_p(buf227.data_ptr()), c_void_p(squeeze_37.data_ptr()))
    del squeeze_37
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf229 = aten.convolution_backward(buf228, mul_97, primals_84, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf228
    del mul_97
    del primals_84
    buf230 = buf229[0]
    buf231 = buf229[1]
    del buf229
    buf232 = reinterpret_tensor(buf201, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf201  # reuse
    buf233 = reinterpret_tensor(buf232, (8, 1, 128), (128, 128, 1), 0); del buf232  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_42(c_void_p(buf233.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(add_61.data_ptr()), c_void_p(convolution_14.data_ptr()))
    # Source Nodes: [sigmoid_2], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf234 = aten.convolution_backward(buf233, view_4, primals_83, [0], [1], [2], [1], False, [0], 1, [True, True, False])
    del buf233
    del primals_83
    del view_4
    buf235 = buf234[0]
    buf236 = buf234[1]
    del buf234
    buf237 = buf212; del buf212  # reuse
    buf238 = empty((128, ), device='cpu', dtype=torch.float32)
    buf239 = buf230; del buf230  # reuse
    buf240 = buf238; del buf238  # reuse
    buf241 = buf239; del buf239  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_43(c_void_p(buf241.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(add_61.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_360.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf237.data_ptr()))
    del add_61
    del buf235
    del convolution_13
    del convolution_14
    del primals_23
    del squeeze_34
    del unsqueeze_360
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf242 = aten.convolution_backward(buf241, mul_88, primals_82, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf241
    del mul_88
    del primals_82
    buf243 = buf242[0]
    buf244 = buf242[1]
    del buf242
    buf245 = empty((128, ), device='cpu', dtype=torch.float32)
    buf246 = empty((128, ), device='cpu', dtype=torch.float32)
    buf247 = empty((128, ), device='cpu', dtype=torch.float32)
    buf248 = buf243; del buf243  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_44(c_void_p(buf248.data_ptr()), c_void_p(mul_505.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_372.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()))
    del buf246
    del convolution_12
    del mul_505
    del primals_21
    del squeeze_31
    del unsqueeze_372
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf249 = aten.convolution_backward(buf248, mul_80, primals_81, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf248
    del mul_80
    del primals_81
    buf250 = buf249[0]
    buf251 = buf249[1]
    del buf249
    buf252 = buf185; del buf185  # reuse
    buf253 = empty((256, ), device='cpu', dtype=torch.float32)
    buf254 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    buf255 = buf253; del buf253  # reuse
    cpp_fused_add_mul_native_batch_norm_backward_45(c_void_p(buf255.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(mul_517.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_384.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf254.data_ptr()))
    del convolution_11
    del primals_19
    del squeeze_28
    del unsqueeze_384
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf256 = aten.convolution_backward(buf254, mul_72, primals_80, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_72
    del primals_80
    buf257 = buf256[0]
    buf258 = buf256[1]
    del buf256
    buf259 = reinterpret_tensor(buf220, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf220  # reuse
    buf260 = reinterpret_tensor(buf259, (8, 1, 64), (64, 64, 1), 0); del buf259  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_46(c_void_p(buf260.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(add_45.data_ptr()), c_void_p(convolution_10.data_ptr()))
    # Source Nodes: [sigmoid_1], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf261 = aten.convolution_backward(buf260, view_2, primals_79, [0], [1], [1], [1], False, [0], 1, [True, True, False])
    del buf260
    del primals_79
    del view_2
    buf262 = buf261[0]
    buf263 = buf261[1]
    del buf261
    buf264 = empty((64, ), device='cpu', dtype=torch.float32)
    buf265 = empty((64, ), device='cpu', dtype=torch.float32)
    buf266 = buf257; del buf257  # reuse
    buf267 = buf265; del buf265  # reuse
    buf268 = buf266; del buf266  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_47(c_void_p(buf268.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(add_45.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_398.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf264.data_ptr()))
    del add_45
    del convolution_10
    del convolution_9
    del primals_17
    del squeeze_25
    del unsqueeze_398
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf269 = aten.convolution_backward(buf268, mul_63, primals_78, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False])
    del buf268
    del mul_63
    del primals_78
    buf270 = buf269[0]
    buf271 = buf269[1]
    del buf269
    buf272 = empty((64, ), device='cpu', dtype=torch.float32)
    buf273 = empty((64, ), device='cpu', dtype=torch.float32)
    buf274 = empty((64, ), device='cpu', dtype=torch.float32)
    buf275 = buf270; del buf270  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_48(c_void_p(buf275.data_ptr()), c_void_p(mul_545.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_410.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()))
    del convolution_8
    del mul_545
    del primals_15
    del squeeze_22
    del unsqueeze_410
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf276 = aten.convolution_backward(buf275, mul_55, primals_77, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf275
    del mul_55
    del primals_77
    buf277 = buf276[0]
    buf278 = buf276[1]
    del buf276
    buf279 = buf224; del buf224  # reuse
    buf280 = empty((256, ), device='cpu', dtype=torch.float32)
    buf281 = empty((256, ), device='cpu', dtype=torch.float32)
    buf287 = empty((256, ), device='cpu', dtype=torch.float32)
    buf282 = empty((256, ), device='cpu', dtype=torch.float32)
    buf283 = buf254; del buf254  # reuse
    buf289 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_49(c_void_p(buf279.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(mul_517.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(mul_557.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_422.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_434.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf289.data_ptr()))
    del buf250
    del buf277
    del buf279
    del buf281
    del convolution_6
    del convolution_7
    del mul_517
    del mul_557
    del primals_11
    del primals_13
    del squeeze_19
    del unsqueeze_422
    del unsqueeze_434
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf284 = aten.convolution_backward(buf283, getitem_6, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf283
    del primals_76
    buf285 = buf284[0]
    buf286 = buf284[1]
    del buf284
    buf288 = buf287; del buf287  # reuse
    cpp_fused_native_batch_norm_backward_50(c_void_p(buf288.data_ptr()), c_void_p(squeeze_16.data_ptr()))
    del squeeze_16
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf290 = aten.convolution_backward(buf289, mul_40, primals_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf289
    del mul_40
    del primals_75
    buf291 = buf290[0]
    buf292 = buf290[1]
    del buf290
    buf293 = reinterpret_tensor(buf262, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf262  # reuse
    buf294 = reinterpret_tensor(buf293, (8, 1, 64), (64, 64, 1), 0); del buf293  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_51(c_void_p(buf294.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(add_24.data_ptr()), c_void_p(convolution_5.data_ptr()))
    # Source Nodes: [sigmoid], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf295 = aten.convolution_backward(buf294, view, primals_74, [0], [1], [1], [1], False, [0], 1, [True, True, False])
    del buf294
    del primals_74
    del view
    buf296 = buf295[0]
    buf297 = buf295[1]
    del buf295
    buf298 = buf273; del buf273  # reuse
    buf299 = empty((64, ), device='cpu', dtype=torch.float32)
    buf300 = buf291; del buf291  # reuse
    buf301 = buf299; del buf299  # reuse
    buf302 = buf300; del buf300  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_52(c_void_p(buf302.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(add_24.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_448.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf298.data_ptr()))
    del add_24
    del buf296
    del convolution_4
    del convolution_5
    del primals_9
    del squeeze_13
    del unsqueeze_448
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf303 = aten.convolution_backward(buf302, mul_31, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False])
    del buf302
    del mul_31
    del primals_73
    buf304 = buf303[0]
    buf305 = buf303[1]
    del buf303
    buf306 = empty((64, ), device='cpu', dtype=torch.float32)
    buf307 = empty((64, ), device='cpu', dtype=torch.float32)
    buf308 = empty((64, ), device='cpu', dtype=torch.float32)
    buf309 = buf304; del buf304  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_53(c_void_p(buf309.data_ptr()), c_void_p(mul_594.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_460.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()))
    del convolution_3
    del mul_594
    del primals_7
    del squeeze_10
    del unsqueeze_460
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf310 = aten.convolution_backward(buf309, getitem_6, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf309
    del getitem_6
    del primals_72
    buf311 = buf310[0]
    buf312 = buf310[1]
    del buf310
    buf313 = buf285; del buf285  # reuse
    cpp_fused_add_54(c_void_p(buf313.data_ptr()), c_void_p(buf311.data_ptr()))
    del buf311
    # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
    buf314 = aten.max_pool2d_with_indices_backward(buf313, mul_23, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_7)
    del buf313
    del getitem_7
    del mul_23
    buf315 = buf314
    del buf314
    buf316 = buf307; del buf307  # reuse
    buf317 = empty((64, ), device='cpu', dtype=torch.float32)
    buf318 = empty((64, ), device='cpu', dtype=torch.float32)
    buf319 = buf315; del buf315  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_55(c_void_p(buf319.data_ptr()), c_void_p(mul_606.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_472.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()))
    del buf317
    del convolution_2
    del mul_606
    del primals_5
    del squeeze_7
    del unsqueeze_472
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf320 = aten.convolution_backward(buf319, mul_15, primals_71, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf319
    del mul_15
    del primals_71
    buf321 = buf320[0]
    buf322 = buf320[1]
    del buf320
    buf323 = empty((32, ), device='cpu', dtype=torch.float32)
    buf324 = empty((32, ), device='cpu', dtype=torch.float32)
    buf325 = empty((32, ), device='cpu', dtype=torch.float32)
    buf326 = buf321; del buf321  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_56(c_void_p(buf326.data_ptr()), c_void_p(mul_618.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_484.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()))
    del buf324
    del convolution_1
    del mul_618
    del primals_3
    del squeeze_4
    del unsqueeze_484
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf327 = aten.convolution_backward(buf326, mul_7, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf326
    del mul_7
    del primals_70
    buf328 = buf327[0]
    buf329 = buf327[1]
    del buf327
    buf330 = empty((24, ), device='cpu', dtype=torch.float32)
    buf331 = empty((24, ), device='cpu', dtype=torch.float32)
    buf332 = empty((24, ), device='cpu', dtype=torch.float32)
    buf333 = buf328; del buf328  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_57(c_void_p(buf333.data_ptr()), c_void_p(mul_630.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_496.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()))
    del buf331
    del convolution
    del mul_630
    del primals_1
    del squeeze_1
    del unsqueeze_496
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf334 = aten.convolution_backward(buf333, primals_203, primals_69, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf333
    del primals_203
    del primals_69
    buf335 = buf334[1]
    return (buf332, buf330, buf325, buf323, buf318, buf316, buf308, buf306, buf301, buf298, buf288, buf280, buf282, buf280, buf274, buf272, buf267, buf264, buf255, buf252, buf247, buf245, buf240, buf237, buf227, buf219, buf221, buf219, buf213, buf211, buf206, buf203, buf194, buf191, buf186, buf184, buf179, buf176, buf166, buf158, buf160, buf158, buf153, buf150, reinterpret_tensor(buf129, (23, 16), (16, 1), 0), reinterpret_tensor(buf125, (23, 16), (16, 1), 0), buf118, buf116, buf112, buf109, buf105, buf102, reinterpret_tensor(buf80, (23, 16), (16, 1), 0), reinterpret_tensor(buf76, (23, 16), (16, 1), 0), buf69, buf67, buf62, buf52, buf55, buf52, buf48, buf45, reinterpret_tensor(buf23, (23, 16), (16, 1), 0), reinterpret_tensor(buf19, (23, 16), (16, 1), 0), buf12, buf10, buf5, buf3, buf335, buf329, buf322, buf312, buf305, buf297, buf292, buf286, buf278, buf271, buf263, buf258, buf251, buf244, buf236, buf231, buf225, buf217, buf210, buf202, buf197, buf190, buf183, buf175, buf170, buf164, buf156, buf149, buf145, buf115, buf108, buf101, buf97, buf66, buf59, buf51, buf44, buf40, buf9, reinterpret_tensor(buf1, (1000, 2048), (2048, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((24, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((32, 24, 3, 3), (216, 1, 72, 24), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((64, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((1, 1, 3), (3, 3, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((64, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((1, 1, 3), (3, 3, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((1, 1, 5), (5, 5, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((128, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((1, 1, 5), (5, 5, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((256, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1, 1, 5), (5, 5, 1), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((384, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((640, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((640, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 24, 128, 128), (393216, 1, 3072, 24), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    mul_7 = rand_strided((8, 24, 128, 128), (393216, 1, 3072, 24), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    mul_15 = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    mul_23 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    getitem_6 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.int64)
    convolution_3 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    mul_31 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_24 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    view = rand_strided((8, 1, 64), (64, 64, 1), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 1, 64), (64, 64, 1), device='cpu', dtype=torch.float32)
    mul_40 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    mul_55 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    mul_63 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_45 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    view_2 = rand_strided((8, 1, 64), (64, 64, 1), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 1, 64), (64, 64, 1), device='cpu', dtype=torch.float32)
    mul_72 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    mul_80 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    mul_88 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_61 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    view_4 = rand_strided((8, 1, 128), (128, 128, 1), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 1, 128), (128, 128, 1), device='cpu', dtype=torch.float32)
    mul_97 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    mul_112 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    mul_120 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_82 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    view_6 = rand_strided((8, 1, 128), (128, 128, 1), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 1, 128), (128, 128, 1), device='cpu', dtype=torch.float32)
    mul_129 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    mul_137 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    mul_145 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    add_98 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    view_8 = rand_strided((8, 1, 256), (256, 256, 1), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 1, 256), (256, 256, 1), device='cpu', dtype=torch.float32)
    mul_154 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    mul_169 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    mul_177 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    view_17 = rand_strided((16384, 16), (16, 1), device='cpu', dtype=torch.float32)
    view_23 = rand_strided((16384, 16), (16, 1), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    mul_186 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    mul_194 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    mul_202 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    view_42 = rand_strided((4096, 16), (16, 1), device='cpu', dtype=torch.float32)
    view_48 = rand_strided((4096, 16), (16, 1), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    mul_211 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    mul_226 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    mul_234 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    view_67 = rand_strided((4096, 16), (16, 1), device='cpu', dtype=torch.float32)
    view_73 = rand_strided((4096, 16), (16, 1), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    mul_243 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    clone_51 = rand_strided((8, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_34 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_253 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    unsqueeze_126 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_265 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    sub_40 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    permute_42 = rand_strided((64, 144, 64), (9216, 1, 144), device='cpu', dtype=torch.float32)
    permute_43 = rand_strided((64, 64, 144), (64, 1, 4096), device='cpu', dtype=torch.float32)
    alias_8 = rand_strided((64, 1, 64, 144), (9216, 9216, 144, 1), device='cpu', dtype=torch.float32)
    permute_47 = rand_strided((23, 16), (16, 1), device='cpu', dtype=torch.float32)
    permute_53 = rand_strided((23, 16), (16, 1), device='cpu', dtype=torch.float32)
    permute_55 = rand_strided((64, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    permute_56 = rand_strided((64, 144, 16), (16, 1024, 1), device='cpu', dtype=torch.float32)
    mul_280 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    unsqueeze_150 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_292 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    unsqueeze_162 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_174 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_313 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    sub_60 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    permute_68 = rand_strided((256, 144, 16), (2304, 1, 144), device='cpu', dtype=torch.float32)
    permute_69 = rand_strided((256, 64, 144), (9216, 1, 64), device='cpu', dtype=torch.float32)
    alias_9 = rand_strided((64, 4, 16, 144), (9216, 1, 576, 4), device='cpu', dtype=torch.float32)
    permute_73 = rand_strided((23, 16), (16, 1), device='cpu', dtype=torch.float32)
    permute_79 = rand_strided((23, 16), (16, 1), device='cpu', dtype=torch.float32)
    permute_81 = rand_strided((256, 16, 16), (256, 1, 16), device='cpu', dtype=torch.float32)
    permute_82 = rand_strided((256, 144, 16), (2304, 1, 144), device='cpu', dtype=torch.float32)
    mul_328 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    unsqueeze_198 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_340 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    unsqueeze_210 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_352 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    sub_76 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    permute_94 = rand_strided((256, 144, 64), (9216, 1, 144), device='cpu', dtype=torch.float32)
    permute_95 = rand_strided((256, 32, 144), (4608, 1, 32), device='cpu', dtype=torch.float32)
    alias_10 = rand_strided((64, 4, 64, 144), (36864, 1, 576, 4), device='cpu', dtype=torch.float32)
    permute_99 = rand_strided((23, 16), (16, 1), device='cpu', dtype=torch.float32)
    permute_105 = rand_strided((23, 16), (16, 1), device='cpu', dtype=torch.float32)
    permute_107 = rand_strided((256, 16, 64), (1024, 1, 16), device='cpu', dtype=torch.float32)
    permute_108 = rand_strided((256, 144, 16), (2304, 1, 144), device='cpu', dtype=torch.float32)
    mul_367 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    unsqueeze_234 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_379 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    unsqueeze_246 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_272 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_416 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    unsqueeze_284 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_428 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    unsqueeze_296 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_456 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_468 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_360 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_505 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    unsqueeze_372 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_517 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    unsqueeze_384 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_398 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_545 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    unsqueeze_410 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_557 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    unsqueeze_422 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_434 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_448 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_594 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    unsqueeze_460 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_606 = rand_strided((8, 64, 128, 128), (1048576, 1, 8192, 64), device='cpu', dtype=torch.float32)
    unsqueeze_472 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_618 = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cpu', dtype=torch.float32)
    unsqueeze_484 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_630 = rand_strided((8, 24, 128, 128), (393216, 1, 3072, 24), device='cpu', dtype=torch.float32)
    unsqueeze_496 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_47, primals_49, primals_51, primals_55, primals_57, primals_59, primals_61, primals_65, primals_67, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_203, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, getitem_6, getitem_7, convolution_3, squeeze_10, mul_31, convolution_4, squeeze_13, add_24, view, convolution_5, mul_40, convolution_6, squeeze_16, convolution_7, squeeze_19, mul_55, convolution_8, squeeze_22, mul_63, convolution_9, squeeze_25, add_45, view_2, convolution_10, mul_72, convolution_11, squeeze_28, mul_80, convolution_12, squeeze_31, mul_88, convolution_13, squeeze_34, add_61, view_4, convolution_14, mul_97, convolution_15, squeeze_37, convolution_16, squeeze_40, mul_112, convolution_17, squeeze_43, mul_120, convolution_18, squeeze_46, add_82, view_6, convolution_19, mul_129, convolution_20, squeeze_49, mul_137, convolution_21, squeeze_52, mul_145, convolution_22, squeeze_55, add_98, view_8, convolution_23, mul_154, convolution_24, squeeze_58, convolution_25, squeeze_61, mul_169, convolution_26, squeeze_64, mul_177, view_17, view_23, squeeze_67, mul_186, convolution_29, squeeze_70, mul_194, convolution_30, squeeze_73, mul_202, view_42, view_48, squeeze_76, mul_211, convolution_33, squeeze_79, convolution_34, squeeze_82, mul_226, convolution_35, squeeze_85, mul_234, view_67, view_73, squeeze_88, mul_243, convolution_38, squeeze_91, clone_51, permute_34, mul_253, unsqueeze_126, mul_265, sub_40, permute_42, permute_43, alias_8, permute_47, permute_53, permute_55, permute_56, mul_280, unsqueeze_150, mul_292, unsqueeze_162, unsqueeze_174, mul_313, sub_60, permute_68, permute_69, alias_9, permute_73, permute_79, permute_81, permute_82, mul_328, unsqueeze_198, mul_340, unsqueeze_210, mul_352, sub_76, permute_94, permute_95, alias_10, permute_99, permute_105, permute_107, permute_108, mul_367, unsqueeze_234, mul_379, unsqueeze_246, unsqueeze_258, unsqueeze_272, mul_416, unsqueeze_284, mul_428, unsqueeze_296, unsqueeze_310, mul_456, unsqueeze_322, mul_468, unsqueeze_334, unsqueeze_346, unsqueeze_360, mul_505, unsqueeze_372, mul_517, unsqueeze_384, unsqueeze_398, mul_545, unsqueeze_410, mul_557, unsqueeze_422, unsqueeze_434, unsqueeze_448, mul_594, unsqueeze_460, mul_606, unsqueeze_472, mul_618, unsqueeze_484, mul_630, unsqueeze_496, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('eca_halonext26ts', benchmark_compiled_module)
