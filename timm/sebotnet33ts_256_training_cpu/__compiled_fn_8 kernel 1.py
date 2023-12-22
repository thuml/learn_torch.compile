
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1280L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1280L*x2) + (81920L*x1)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1280L*x2) + (81920L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1280L); x2+=static_cast<long>(8L))
                    {
                        float tmp24[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1280L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1280L*x1) + (1280L*x1_inner) + (81920L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1280L*x1) + (1280L*x1_inner) + (81920L*x0)));
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
                        at::vec::transpose_mxn<float,8,8>(tmp24, 8, out_ptr4 + static_cast<long>(x1 + (64L*x2) + (81920L*x0)), static_cast<long>(64L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1536L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1536L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.001953125);
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
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mul_native_batch_norm_backward_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x3) + (4096L*x2) + (32768L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (4096L*x2) + (32768L*x1)));
                                auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr2[static_cast<long>((128L*x3) + (1024L*x2) + (8192L*(c10::div_floor_integer((x3 + (8L*x2) + (64L*x0) + (64L*x0_inner)), 8192L))) + (32768L*x1) + (static_cast<long>(c10::div_floor_integer((x3 + (8L*x2) + (64L*x0) + (64L*x0_inner)), 64L)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = tmp3 - tmp4;
                                auto tmp6 = tmp2 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (32768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (32768L*x0)));
                        auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((128L*x1) + (8192L*(c10::div_floor_integer((x1 + (64L*x2) + (64L*x2_inner)), 8192L))) + (32768L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp7 = static_cast<float>(0.001953125);
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp20.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr3[static_cast<long>(x1 + (64L*x2) + (64L*x2_inner) + (32768L*x0))] = tmpbuf[x2_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_view_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                        {
                            {
                                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                                float tmp_acc0 = 0;
                                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                                for(long x4=static_cast<long>(0L); x4<static_cast<long>(8L); x4+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x4 + (8L*x3) + (64L*x2) + (512L*x1) + (4096L*x0)));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x4 + (8L*x3) + (64L*x2) + (512L*x1) + (4096L*x0)));
                                    auto tmp3 = out_ptr0[static_cast<long>(x2 + (8L*x1) + (64L*x0))];
                                    auto tmp2 = tmp0 * tmp1;
                                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                    auto tmp5 = tmp1 * tmp4;
                                    auto tmp6 = tmp2 - tmp5;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                }
                                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                                out_ptr1[static_cast<long>(x3 + (8L*x1) + (64L*x2) + (512L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(15L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(16);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L))));
                            auto tmp5 = static_cast<long>(135);
                            auto tmp6 = tmp4 < tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L)))), 15L)) % static_cast<long>(9L));
                                auto tmp9 = static_cast<long>(8);
                                auto tmp10 = tmp8 < tmp9;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L))))) % static_cast<long>(15L));
                                    auto tmp13 = static_cast<long>(7);
                                    auto tmp14 = tmp12 >= tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp16 = out_ptr1[static_cast<long>((-7L) + (8L*(static_cast<long>(c10::div_floor_integer((x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L)))), 15L)) % static_cast<long>(9L))) + (64L*(c10::div_floor_integer(x0, 8L))) + (static_cast<long>((x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L))))) % static_cast<long>(15L)))];
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
                        out_ptr2[static_cast<long>(x1 + (15L*x0))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (8L*x2) + (64L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (8L*x2) + (64L*x0)));
                            auto tmp3 = in_ptr2[static_cast<long>(x0)];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (8L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(15L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(16);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L))));
                            auto tmp5 = static_cast<long>(135);
                            auto tmp6 = tmp4 < tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L)))), 15L)) % static_cast<long>(9L));
                                auto tmp9 = static_cast<long>(8);
                                auto tmp10 = tmp8 < tmp9;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L))))) % static_cast<long>(15L));
                                    auto tmp13 = static_cast<long>(7);
                                    auto tmp14 = tmp12 >= tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp16 = out_ptr0[static_cast<long>((-7L) + (8L*(static_cast<long>(c10::div_floor_integer((x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L)))), 15L)) % static_cast<long>(9L))) + (64L*(c10::div_floor_integer(x0, 8L))) + (static_cast<long>((x1 + (16L*(static_cast<long>(x0) % static_cast<long>(8L))))) % static_cast<long>(15L)))];
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
                        out_ptr1[static_cast<long>(x1 + (15L*x0))] = tmp24;
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
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp3 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(0.08838834764831845);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_6 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1536L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x3);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(512);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((128L*x1) + (1024L*x2) + (1024L*x2_inner) + (8192L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3) + (32768L*x0)), 8192L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 64L)) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((128L*x2) + (128L*x2_inner) + (1024L*x1) + (8192L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3) + (32768L*x0)), 8192L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 64L)) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp8 = tmp6 + tmp7;
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((128L*x2) + (128L*x2_inner) + (1024L*x1) + (8192L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3) + (32768L*x0)), 8192L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 64L)) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp10 = tmp8 + tmp9;
                                return tmp10;
                            }
                            ;
                            auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp12 = tmp0 >= tmp3;
                            auto tmp13 = static_cast<int>(1024);
                            auto tmp14 = tmp0 < tmp13;
                            auto tmp15 = tmp12 & tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr3 + static_cast<long>((-32768L) + x2 + (8L*x1) + (64L*x3) + (32768L*x0)), to_float_mask(tmp15));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp15));
                            auto tmp19 = tmp0 >= tmp13;
                            auto tmp20 = static_cast<int>(1536);
                            auto tmp21 = tmp0 < tmp20;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr4[static_cast<long>((128L*x2) + (128L*x2_inner) + (1024L*x1) + (8192L*(static_cast<long>(c10::div_floor_integer(((-65536L) + x2 + x2_inner + (8L*x1) + (64L*x3) + (32768L*x0)), 8192L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 64L)) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp19)); })();
                                return tmp23;
                            }
                            ;
                            auto tmp24 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp19));
                            auto tmp25 = to_float_mask(tmp15);
                            auto tmp26 = decltype(tmp18)::blendv(tmp24, tmp18, tmp25);
                            auto tmp27 = to_float_mask(tmp4);
                            auto tmp28 = decltype(tmp11)::blendv(tmp26, tmp11, tmp27);
                            tmp28.store(out_ptr0 + static_cast<long>(x2 + (8L*x1) + (64L*x3) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_7 = async_compile.cpp('''
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.001953125);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
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
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr2;
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1536L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1536L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1536L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1536L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1536L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (1536L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp9 = tmp7 - tmp8;
                        auto tmp10 = tmp6 * tmp9;
                        auto tmp13 = tmp11 - tmp12;
                        auto tmp14 = tmp6 * tmp13;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                        tmp_acc2_vec = tmp_acc2_vec + tmp14;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp36 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp11 = static_cast<float>(0.001953125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp6 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp24 = tmp22 - tmp23;
                    auto tmp26 = tmp25 * tmp12;
                    auto tmp28 = tmp27 * tmp27;
                    auto tmp29 = tmp26 * tmp28;
                    auto tmp30 = tmp24 * tmp29;
                    auto tmp31 = tmp6 - tmp30;
                    auto tmp32 = tmp31 - tmp20;
                    auto tmp34 = tmp14 * tmp33;
                    auto tmp35 = tmp21 * tmp34;
                    auto tmp37 = tmp27 * tmp36;
                    auto tmp38 = tmp32 * tmp37;
                    tmp35.store(in_out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    tmp38.store(in_out_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_avg_pool2d_backward_mul_native_batch_norm_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.001953125);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*(std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(8L, 1L + (c10::div_floor_integer(x3, 2L))))))) + (512L*(((std::min(std::max(0L, c10::div_floor_integer(x3, 2L)), (-1L) + (std::min(8L, 1L + (c10::div_floor_integer(x3, 2L)))))) >= 0L) ? 0L : 8L)) + (4096L*(std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(8L, 1L + (c10::div_floor_integer(x2, 2L))))))) + (4096L*(((std::min(std::max(0L, c10::div_floor_integer(x2, 2L)), (-1L) + (std::min(8L, 1L + (c10::div_floor_integer(x2, 2L)))))) >= 0L) ? 0L : 8L)) + (32768L*x0))];
                            auto tmp1 = tmp0 / 4;
                            auto tmp2 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x2, 2L)));
                            auto tmp3 = c10::convert<int>(std::min(8L, 1L + (c10::div_floor_integer(x2, 2L))));
                            auto tmp4 = tmp2 < tmp3;
                            auto tmp5 = c10::convert<int>(std::max(0L, c10::div_floor_integer(x3, 2L)));
                            auto tmp6 = c10::convert<int>(std::min(8L, 1L + (c10::div_floor_integer(x3, 2L))));
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = tmp4 & tmp7;
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = tmp8 ? tmp1 : tmp9;
                            out_ptr3[static_cast<long>(x3 + (16L*x2) + (256L*x1) + (131072L*x0))] = tmp10;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_view_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            {
                                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                                float tmp_acc0 = 0;
                                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                                for(long x4=static_cast<long>(0L); x4<static_cast<long>(16L); x4+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x4 + (16L*x3) + (256L*x2) + (4096L*x1) + (65536L*x0)));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x4 + (16L*x3) + (256L*x2) + (4096L*x1) + (65536L*x0)));
                                    auto tmp3 = out_ptr0[static_cast<long>(x2 + (16L*x1) + (256L*x0))];
                                    auto tmp2 = tmp0 * tmp1;
                                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                    auto tmp5 = tmp1 * tmp4;
                                    auto tmp6 = tmp2 - tmp5;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                }
                                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                                out_ptr1[static_cast<long>(x3 + (16L*x1) + (256L*x2) + (4096L*x0))] = static_cast<float>(tmp_acc0);
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(31L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(32);
                    auto tmp2 = tmp0 < tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<long>(x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))));
                        auto tmp5 = static_cast<long>(527);
                        auto tmp6 = tmp4 < tmp5;
                        auto tmp7 = [&]
                        {
                            auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L)))), 31L)) % static_cast<long>(17L));
                            auto tmp9 = static_cast<long>(16);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))))) % static_cast<long>(31L));
                                auto tmp13 = static_cast<long>(15);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = [&]
                                {
                                    auto tmp16 = out_ptr1[static_cast<long>((-15L) + (16L*(static_cast<long>(c10::div_floor_integer((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L)))), 31L)) % static_cast<long>(17L))) + (256L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))))) % static_cast<long>(31L)))];
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
                    out_ptr2[static_cast<long>(x1 + (31L*x0))] = tmp24;
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x2) + (256L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x2) + (256L*x0)));
                            auto tmp3 = in_ptr2[static_cast<long>(x0)];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(31L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(32);
                    auto tmp2 = tmp0 < tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<long>(x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))));
                        auto tmp5 = static_cast<long>(527);
                        auto tmp6 = tmp4 < tmp5;
                        auto tmp7 = [&]
                        {
                            auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L)))), 31L)) % static_cast<long>(17L));
                            auto tmp9 = static_cast<long>(16);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))))) % static_cast<long>(31L));
                                auto tmp13 = static_cast<long>(15);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = [&]
                                {
                                    auto tmp16 = out_ptr0[static_cast<long>((-15L) + (16L*(static_cast<long>(c10::div_floor_integer((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L)))), 31L)) % static_cast<long>(17L))) + (256L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))))) % static_cast<long>(31L)))];
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
                    out_ptr1[static_cast<long>(x1 + (31L*x0))] = tmp24;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(0.08838834764831845);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_14 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1536L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x3);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(512);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((128L*x1) + (2048L*x2) + (2048L*x2_inner) + (32768L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3) + (131072L*x0)), 32768L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((128L*x2) + (128L*x2_inner) + (2048L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3) + (131072L*x0)), 32768L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp8 = tmp6 + tmp7;
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((128L*x2) + (128L*x2_inner) + (2048L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3) + (131072L*x0)), 32768L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp10 = tmp8 + tmp9;
                                return tmp10;
                            }
                            ;
                            auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp12 = tmp0 >= tmp3;
                            auto tmp13 = static_cast<int>(1024);
                            auto tmp14 = tmp0 < tmp13;
                            auto tmp15 = tmp12 & tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr3 + static_cast<long>((-131072L) + x2 + (16L*x1) + (256L*x3) + (131072L*x0)), to_float_mask(tmp15));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp15));
                            auto tmp19 = tmp0 >= tmp13;
                            auto tmp20 = static_cast<int>(1536);
                            auto tmp21 = tmp0 < tmp20;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr4[static_cast<long>((128L*x2) + (128L*x2_inner) + (2048L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(((-262144L) + x2 + x2_inner + (16L*x1) + (256L*x3) + (131072L*x0)), 32768L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp19)); })();
                                return tmp23;
                            }
                            ;
                            auto tmp24 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp19));
                            auto tmp25 = to_float_mask(tmp15);
                            auto tmp26 = decltype(tmp18)::blendv(tmp24, tmp18, tmp25);
                            auto tmp27 = to_float_mask(tmp4);
                            auto tmp28 = decltype(tmp11)::blendv(tmp26, tmp11, tmp27);
                            tmp28.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (256L*x3) + (393216L*x0)));
                        }
                    }
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.00048828125);
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
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_batch_norm_backward_16 = async_compile.cpp('''
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


cpp_fused_mul_native_batch_norm_backward_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x3) + (4096L*x2) + (65536L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x3) + (4096L*x2) + (65536L*x1)));
                                auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr2[static_cast<long>((64L*x3) + (1024L*x2) + (16384L*(c10::div_floor_integer((x3 + (16L*x2) + (256L*x0) + (256L*x0_inner)), 16384L))) + (65536L*x1) + (static_cast<long>(c10::div_floor_integer((x3 + (16L*x2) + (256L*x0) + (256L*x0_inner)), 256L)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = tmp3 - tmp4;
                                auto tmp6 = tmp2 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((64L*x1) + (16384L*(c10::div_floor_integer((x1 + (256L*x2) + (256L*x2_inner)), 16384L))) + (65536L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp7 = static_cast<float>(0.00048828125);
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp20.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr3[static_cast<long>(x1 + (256L*x2) + (256L*x2_inner) + (65536L*x0))] = tmpbuf[x2_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_view_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            {
                                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                                float tmp_acc0 = 0;
                                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                                for(long x4=static_cast<long>(0L); x4<static_cast<long>(16L); x4+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x4 + (16L*x3) + (256L*x2) + (4096L*x1) + (65536L*x0)));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x4 + (16L*x3) + (256L*x2) + (4096L*x1) + (65536L*x0)));
                                    auto tmp3 = out_ptr0[static_cast<long>(x2 + (16L*x1) + (256L*x0))];
                                    auto tmp2 = tmp0 * tmp1;
                                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                    auto tmp5 = tmp1 * tmp4;
                                    auto tmp6 = tmp2 - tmp5;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                }
                                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                                out_ptr1[static_cast<long>(x3 + (16L*x1) + (256L*x2) + (4096L*x0))] = static_cast<float>(tmp_acc0);
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(31L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(32);
                    auto tmp2 = tmp0 < tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<long>(x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))));
                        auto tmp5 = static_cast<long>(527);
                        auto tmp6 = tmp4 < tmp5;
                        auto tmp7 = [&]
                        {
                            auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L)))), 31L)) % static_cast<long>(17L));
                            auto tmp9 = static_cast<long>(16);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))))) % static_cast<long>(31L));
                                auto tmp13 = static_cast<long>(15);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = [&]
                                {
                                    auto tmp16 = out_ptr1[static_cast<long>((-15L) + (16L*(static_cast<long>(c10::div_floor_integer((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L)))), 31L)) % static_cast<long>(17L))) + (256L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))))) % static_cast<long>(31L)))];
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
                    out_ptr2[static_cast<long>(x1 + (31L*x0))] = tmp24;
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x2) + (256L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x2) + (256L*x0)));
                            auto tmp3 = in_ptr2[static_cast<long>(x0)];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(31L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(32);
                    auto tmp2 = tmp0 < tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<long>(x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))));
                        auto tmp5 = static_cast<long>(527);
                        auto tmp6 = tmp4 < tmp5;
                        auto tmp7 = [&]
                        {
                            auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L)))), 31L)) % static_cast<long>(17L));
                            auto tmp9 = static_cast<long>(16);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))))) % static_cast<long>(31L));
                                auto tmp13 = static_cast<long>(15);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = [&]
                                {
                                    auto tmp16 = out_ptr0[static_cast<long>((-15L) + (16L*(static_cast<long>(c10::div_floor_integer((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L)))), 31L)) % static_cast<long>(17L))) + (256L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + (32L*(static_cast<long>(x0) % static_cast<long>(16L))))) % static_cast<long>(31L)))];
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
                    out_ptr1[static_cast<long>(x1 + (31L*x0))] = tmp24;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(0.125);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_21 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x3);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(256);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((64L*x1) + (1024L*x2) + (1024L*x2_inner) + (16384L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3) + (65536L*x0)), 16384L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((64L*x2) + (64L*x2_inner) + (1024L*x1) + (16384L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3) + (65536L*x0)), 16384L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp8 = tmp6 + tmp7;
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((64L*x2) + (64L*x2_inner) + (1024L*x1) + (16384L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3) + (65536L*x0)), 16384L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp10 = tmp8 + tmp9;
                                return tmp10;
                            }
                            ;
                            auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp12 = tmp0 >= tmp3;
                            auto tmp13 = static_cast<int>(512);
                            auto tmp14 = tmp0 < tmp13;
                            auto tmp15 = tmp12 & tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr3 + static_cast<long>((-65536L) + x2 + (16L*x1) + (256L*x3) + (65536L*x0)), to_float_mask(tmp15));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp15));
                            auto tmp19 = tmp0 >= tmp13;
                            auto tmp20 = static_cast<int>(768);
                            auto tmp21 = tmp0 < tmp20;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr4[static_cast<long>((64L*x2) + (64L*x2_inner) + (1024L*x1) + (16384L*(static_cast<long>(c10::div_floor_integer(((-131072L) + x2 + x2_inner + (16L*x1) + (256L*x3) + (65536L*x0)), 16384L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp19)); })();
                                return tmp23;
                            }
                            ;
                            auto tmp24 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp19));
                            auto tmp25 = to_float_mask(tmp15);
                            auto tmp26 = decltype(tmp18)::blendv(tmp24, tmp18, tmp25);
                            auto tmp27 = to_float_mask(tmp4);
                            auto tmp28 = decltype(tmp11)::blendv(tmp26, tmp11, tmp27);
                            tmp28.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (256L*x3) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_22 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(0.00048828125);
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


cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1024L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_24 = async_compile.cpp('''
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


cpp_fused_convolution_backward_threshold_backward_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_26 = async_compile.cpp('''
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


cpp_fused_convolution_backward_mul_native_batch_norm_backward_27 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(0.00048828125);
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


cpp_fused_add_mul_native_batch_norm_backward_28 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp4 * tmp7;
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp4 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                        tmp_acc2_vec = tmp_acc2_vec + tmp12;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
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
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
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
                    auto tmp25 = tmp23 - tmp24;
                    auto tmp27 = tmp26 * tmp10;
                    auto tmp29 = tmp28 * tmp28;
                    auto tmp30 = tmp27 * tmp29;
                    auto tmp31 = tmp25 * tmp30;
                    auto tmp32 = tmp4 - tmp31;
                    auto tmp33 = tmp32 - tmp18;
                    auto tmp35 = tmp28 * tmp34;
                    auto tmp36 = tmp33 * tmp35;
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    tmp36.store(out_ptr4 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_29 = async_compile.cpp('''
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_30 = async_compile.cpp('''
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


cpp_fused_convolution_backward_threshold_backward_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_32 = async_compile.cpp('''
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


cpp_fused_convolution_backward_mul_native_batch_norm_backward_33 = async_compile.cpp('''
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


cpp_fused_add_mul_native_batch_norm_backward_34 = async_compile.cpp('''
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


cpp_fused_mul_native_batch_norm_backward_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x3) + (4096L*x2) + (131072L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x3) + (4096L*x2) + (131072L*x1)));
                                auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr2[static_cast<long>((32L*x3) + (1024L*x2) + (32768L*(c10::div_floor_integer((x3 + (32L*x2) + (1024L*x0) + (1024L*x0_inner)), 32768L))) + (131072L*x1) + (static_cast<long>(c10::div_floor_integer((x3 + (32L*x2) + (1024L*x0) + (1024L*x0_inner)), 1024L)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = tmp3 - tmp4;
                                auto tmp6 = tmp2 * tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (131072L*x0)));
                        auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((32L*x1) + (32768L*(c10::div_floor_integer((x1 + (1024L*x2) + (1024L*x2_inner)), 32768L))) + (131072L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp20.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr3[static_cast<long>(x1 + (1024L*x2) + (1024L*x2_inner) + (131072L*x0))] = tmpbuf[x2_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_view_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(1L))
                        {
                            {
                                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                                float tmp_acc0 = 0;
                                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                                for(long x4=static_cast<long>(0L); x4<static_cast<long>(32L); x4+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x4 + (32L*x3) + (1024L*x2) + (32768L*x1) + (1048576L*x0)));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x4 + (32L*x3) + (1024L*x2) + (32768L*x1) + (1048576L*x0)));
                                    auto tmp3 = out_ptr0[static_cast<long>(x2 + (32L*x1) + (1024L*x0))];
                                    auto tmp2 = tmp0 * tmp1;
                                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                    auto tmp5 = tmp1 * tmp4;
                                    auto tmp6 = tmp2 - tmp5;
                                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                }
                                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                                out_ptr1[static_cast<long>(x3 + (32L*x1) + (1024L*x2) + (32768L*x0))] = static_cast<float>(tmp_acc0);
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(63L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(64);
                    auto tmp2 = tmp0 < tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<long>(x1 + (64L*(static_cast<long>(x0) % static_cast<long>(32L))));
                        auto tmp5 = static_cast<long>(2079);
                        auto tmp6 = tmp4 < tmp5;
                        auto tmp7 = [&]
                        {
                            auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (64L*(static_cast<long>(x0) % static_cast<long>(32L)))), 63L)) % static_cast<long>(33L));
                            auto tmp9 = static_cast<long>(32);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (64L*(static_cast<long>(x0) % static_cast<long>(32L))))) % static_cast<long>(63L));
                                auto tmp13 = static_cast<long>(31);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = [&]
                                {
                                    auto tmp16 = out_ptr1[static_cast<long>((-31L) + (32L*(static_cast<long>(c10::div_floor_integer((x1 + (64L*(static_cast<long>(x0) % static_cast<long>(32L)))), 63L)) % static_cast<long>(33L))) + (1024L*(c10::div_floor_integer(x0, 32L))) + (static_cast<long>((x1 + (64L*(static_cast<long>(x0) % static_cast<long>(32L))))) % static_cast<long>(63L)))];
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
                    out_ptr2[static_cast<long>(x1 + (63L*x0))] = tmp24;
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x2) + (1024L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x2) + (1024L*x0)));
                            auto tmp3 = in_ptr2[static_cast<long>(x0)];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(63L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(64);
                    auto tmp2 = tmp0 < tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = c10::convert<long>(x1 + (64L*(static_cast<long>(x0) % static_cast<long>(32L))));
                        auto tmp5 = static_cast<long>(2079);
                        auto tmp6 = tmp4 < tmp5;
                        auto tmp7 = [&]
                        {
                            auto tmp8 = c10::convert<long>(static_cast<long>(c10::div_floor_integer((x1 + (64L*(static_cast<long>(x0) % static_cast<long>(32L)))), 63L)) % static_cast<long>(33L));
                            auto tmp9 = static_cast<long>(32);
                            auto tmp10 = tmp8 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = c10::convert<long>(static_cast<long>((x1 + (64L*(static_cast<long>(x0) % static_cast<long>(32L))))) % static_cast<long>(63L));
                                auto tmp13 = static_cast<long>(31);
                                auto tmp14 = tmp12 >= tmp13;
                                auto tmp15 = [&]
                                {
                                    auto tmp16 = out_ptr0[static_cast<long>((-31L) + (32L*(static_cast<long>(c10::div_floor_integer((x1 + (64L*(static_cast<long>(x0) % static_cast<long>(32L)))), 63L)) % static_cast<long>(33L))) + (1024L*(c10::div_floor_integer(x0, 32L))) + (static_cast<long>((x1 + (64L*(static_cast<long>(x0) % static_cast<long>(32L))))) % static_cast<long>(63L)))];
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
                    out_ptr1[static_cast<long>(x1 + (63L*x0))] = tmp24;
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = in_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(0.1767766952966369);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_39 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x3);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(128);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((32L*x1) + (1024L*x2) + (1024L*x2_inner) + (32768L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (32L*x1) + (1024L*x3) + (131072L*x0)), 32768L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (32L*x1) + (1024L*x3)), 1024L)) % static_cast<long>(32L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((32L*x2) + (32L*x2_inner) + (1024L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (32L*x1) + (1024L*x3) + (131072L*x0)), 32768L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (32L*x1) + (1024L*x3)), 1024L)) % static_cast<long>(32L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp8 = tmp6 + tmp7;
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((32L*x2) + (32L*x2_inner) + (1024L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (32L*x1) + (1024L*x3) + (131072L*x0)), 32768L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (32L*x1) + (1024L*x3)), 1024L)) % static_cast<long>(32L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp10 = tmp8 + tmp9;
                                return tmp10;
                            }
                            ;
                            auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp12 = tmp0 >= tmp3;
                            auto tmp13 = static_cast<int>(256);
                            auto tmp14 = tmp0 < tmp13;
                            auto tmp15 = tmp12 & tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr3 + static_cast<long>((-131072L) + x2 + (32L*x1) + (1024L*x3) + (131072L*x0)), to_float_mask(tmp15));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp15));
                            auto tmp19 = tmp0 >= tmp13;
                            auto tmp20 = static_cast<int>(384);
                            auto tmp21 = tmp0 < tmp20;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr4[static_cast<long>((32L*x2) + (32L*x2_inner) + (1024L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(((-262144L) + x2 + x2_inner + (32L*x1) + (1024L*x3) + (131072L*x0)), 32768L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (32L*x1) + (1024L*x3)), 1024L)) % static_cast<long>(32L)))]; return masked_load(tmpbuf, to_float_mask(tmp19)); })();
                                return tmp23;
                            }
                            ;
                            auto tmp24 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp19));
                            auto tmp25 = to_float_mask(tmp15);
                            auto tmp26 = decltype(tmp18)::blendv(tmp24, tmp18, tmp25);
                            auto tmp27 = to_float_mask(tmp4);
                            auto tmp28 = decltype(tmp11)::blendv(tmp26, tmp11, tmp27);
                            tmp28.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (1024L*x3) + (393216L*x0)));
                        }
                    }
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


cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_41 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
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
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_convolution_backward_threshold_backward_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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


cpp_fused_add_mul_native_batch_norm_backward_46 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp4 * tmp7;
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp4 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                        tmp_acc2_vec = tmp_acc2_vec + tmp12;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
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
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
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
                    auto tmp25 = tmp23 - tmp24;
                    auto tmp27 = tmp26 * tmp10;
                    auto tmp29 = tmp28 * tmp28;
                    auto tmp30 = tmp27 * tmp29;
                    auto tmp31 = tmp25 * tmp30;
                    auto tmp32 = tmp4 - tmp31;
                    auto tmp33 = tmp32 - tmp18;
                    auto tmp35 = tmp28 * tmp34;
                    auto tmp36 = tmp33 * tmp35;
                    tmp22.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    tmp36.store(out_ptr4 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_47 = async_compile.cpp('''
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_48 = async_compile.cpp('''
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


cpp_fused_convolution_backward_threshold_backward_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_50 = async_compile.cpp('''
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


cpp_fused_convolution_backward_mul_native_batch_norm_backward_51 = async_compile.cpp('''
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


cpp_fused_add_mul_native_batch_norm_backward_52 = async_compile.cpp('''
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_53 = async_compile.cpp('''
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


cpp_fused_convolution_backward_threshold_backward_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_55 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_57 = async_compile.cpp('''
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


cpp_fused_native_batch_norm_backward_58 = async_compile.cpp('''
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


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_59 = async_compile.cpp('''
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


cpp_fused_convolution_backward_threshold_backward_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_61 = async_compile.cpp('''
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


cpp_fused_convolution_backward_mul_native_batch_norm_backward_62 = async_compile.cpp('''
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


cpp_fused_add_mul_native_batch_norm_backward_63 = async_compile.cpp('''
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_64 = async_compile.cpp('''
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
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_61, primals_63, primals_65, primals_69, primals_71, primals_73, primals_75, primals_79, primals_81, primals_83, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_92, primals_94, primals_95, primals_96, primals_97, primals_98, primals_100, primals_102, primals_103, primals_104, primals_105, primals_107, primals_109, primals_110, primals_111, primals_112, primals_113, primals_115, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_125, primals_127, primals_128, primals_129, primals_130, primals_131, primals_133, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_263, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, convolution_3, squeeze_10, mul_31, convolution_4, squeeze_13, add_24, mean, relu, convolution_6, mul_40, convolution_7, squeeze_16, convolution_8, squeeze_19, mul_55, convolution_9, squeeze_22, mul_63, convolution_10, squeeze_25, add_45, mean_1, relu_1, convolution_12, mul_72, convolution_13, squeeze_28, mul_80, convolution_14, squeeze_31, mul_88, convolution_15, squeeze_34, add_61, mean_2, relu_2, convolution_17, mul_97, convolution_18, squeeze_37, convolution_19, squeeze_40, mul_112, convolution_20, squeeze_43, mul_120, convolution_21, squeeze_46, add_82, mean_3, relu_3, convolution_23, mul_129, convolution_24, squeeze_49, mul_137, convolution_25, squeeze_52, mul_145, view_7, view_13, bmm_1, squeeze_55, mul_154, convolution_27, squeeze_58, mul_162, convolution_28, squeeze_61, mul_170, convolution_29, squeeze_64, add_116, mean_4, relu_4, convolution_31, mul_179, convolution_32, squeeze_67, convolution_33, squeeze_70, mul_194, convolution_34, squeeze_73, mul_202, convolution_35, squeeze_76, add_137, mean_5, relu_5, convolution_37, mul_211, convolution_38, squeeze_79, mul_219, convolution_39, squeeze_82, mul_227, view_31, view_37, bmm_3, squeeze_85, mul_236, convolution_41, squeeze_88, mul_244, convolution_42, squeeze_91, mul_252, view_55, view_61, view_71, avg_pool2d, squeeze_94, mul_261, convolution_44, squeeze_97, convolution_45, squeeze_100, mul_276, convolution_46, squeeze_103, mul_284, view_79, view_85, bmm_7, squeeze_106, mul_293, convolution_48, squeeze_109, mul_301, convolution_49, squeeze_112, clone_62, permute_33, mul_311, unsqueeze_154, mul_323, unsqueeze_166, mul_335, unsqueeze_178, permute_41, permute_42, alias_16, permute_46, permute_52, permute_54, permute_55, mul_350, unsqueeze_190, mul_362, unsqueeze_202, unsqueeze_214, mul_383, unsqueeze_226, permute_62, permute_63, alias_17, permute_67, permute_73, permute_75, permute_76, mul_398, unsqueeze_238, mul_410, unsqueeze_250, mul_422, unsqueeze_262, permute_83, permute_84, alias_18, permute_88, permute_94, permute_96, permute_97, mul_437, unsqueeze_274, mul_449, unsqueeze_286, unsqueeze_298, mul_477, unsqueeze_310, mul_489, unsqueeze_322, unsqueeze_334, unsqueeze_346, mul_526, unsqueeze_358, mul_538, unsqueeze_370, mul_550, unsqueeze_382, permute_110, permute_111, alias_27, permute_115, permute_121, permute_123, permute_124, mul_565, unsqueeze_394, mul_577, unsqueeze_406, unsqueeze_418, mul_605, unsqueeze_430, mul_617, unsqueeze_442, unsqueeze_454, unsqueeze_466, mul_654, unsqueeze_478, mul_666, unsqueeze_490, unsqueeze_502, mul_694, unsqueeze_514, mul_706, unsqueeze_526, unsqueeze_538, unsqueeze_550, mul_743, unsqueeze_562, mul_755, unsqueeze_574, mul_767, unsqueeze_586, mul_779, unsqueeze_598, tangents_1 = args
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
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_45, (256, ), (1, ))
    assert_size_stride(primals_47, (1024, ), (1, ))
    assert_size_stride(primals_49, (1024, ), (1, ))
    assert_size_stride(primals_51, (256, ), (1, ))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_55, (1024, ), (1, ))
    assert_size_stride(primals_57, (256, ), (1, ))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_63, (1024, ), (1, ))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_69, (512, ), (1, ))
    assert_size_stride(primals_71, (1536, ), (1, ))
    assert_size_stride(primals_73, (1536, ), (1, ))
    assert_size_stride(primals_75, (512, ), (1, ))
    assert_size_stride(primals_79, (512, ), (1, ))
    assert_size_stride(primals_81, (1536, ), (1, ))
    assert_size_stride(primals_83, (1280, ), (1, ))
    assert_size_stride(primals_85, (24, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_86, (32, 24, 3, 3), (216, 1, 72, 24))
    assert_size_stride(primals_87, (64, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_88, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_89, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_90, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_92, (64, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_94, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_95, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_96, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_97, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_98, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_100, (64, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_102, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_103, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_104, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_105, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_107, (128, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_109, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_110, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_111, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_112, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_113, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_115, (128, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_117, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_118, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_119, (384, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_120, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_121, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_122, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_123, (16, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_125, (256, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_127, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_128, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_129, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_130, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_131, (16, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_133, (256, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_135, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_136, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_137, (768, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_138, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_139, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_140, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_141, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_142, (1536, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_143, (512, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_144, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_145, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_146, (1280, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_263, (8, 3, 256, 256), (196608, 1, 768, 3))
    assert_size_stride(convolution, (8, 24, 128, 128), (393216, 1, 3072, 24))
    assert_size_stride(squeeze_1, (24, ), (1, ))
    assert_size_stride(mul_7, (8, 24, 128, 128), (393216, 1, 3072, 24))
    assert_size_stride(convolution_1, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(mul_15, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(convolution_2, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(mul_23, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_3, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_10, (64, ), (1, ))
    assert_size_stride(mul_31, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_4, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_13, (64, ), (1, ))
    assert_size_stride(add_24, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(mean, (8, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(relu, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(convolution_6, (8, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(mul_40, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_7, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(squeeze_16, (256, ), (1, ))
    assert_size_stride(convolution_8, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(squeeze_19, (256, ), (1, ))
    assert_size_stride(mul_55, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(convolution_9, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_22, (64, ), (1, ))
    assert_size_stride(mul_63, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_10, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_25, (64, ), (1, ))
    assert_size_stride(add_45, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(mean_1, (8, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(relu_1, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(convolution_12, (8, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(mul_72, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_13, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(squeeze_28, (256, ), (1, ))
    assert_size_stride(mul_80, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(convolution_14, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(squeeze_31, (128, ), (1, ))
    assert_size_stride(mul_88, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(convolution_15, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_34, (128, ), (1, ))
    assert_size_stride(add_61, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(mean_2, (8, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(relu_2, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(convolution_17, (8, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(mul_97, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_18, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(squeeze_37, (512, ), (1, ))
    assert_size_stride(convolution_19, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(squeeze_40, (512, ), (1, ))
    assert_size_stride(mul_112, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(convolution_20, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_43, (128, ), (1, ))
    assert_size_stride(mul_120, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_21, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_46, (128, ), (1, ))
    assert_size_stride(add_82, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(mean_3, (8, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(relu_3, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(convolution_23, (8, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(mul_129, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_24, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(squeeze_49, (512, ), (1, ))
    assert_size_stride(mul_137, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(convolution_25, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_52, (128, ), (1, ))
    assert_size_stride(mul_145, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(view_7, (32768, 32), (32, 1))
    assert_size_stride(view_13, (32768, 32), (32, 1))
    assert_size_stride(bmm_1, (32, 1024, 32), (32768, 32, 1))
    assert_size_stride(squeeze_55, (128, ), (1, ))
    assert_size_stride(mul_154, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_27, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(squeeze_58, (512, ), (1, ))
    assert_size_stride(mul_162, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(convolution_28, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(squeeze_61, (256, ), (1, ))
    assert_size_stride(mul_170, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(convolution_29, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_64, (256, ), (1, ))
    assert_size_stride(add_116, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(mean_4, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(relu_4, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(convolution_31, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(mul_179, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_32, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(squeeze_67, (1024, ), (1, ))
    assert_size_stride(convolution_33, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(squeeze_70, (1024, ), (1, ))
    assert_size_stride(mul_194, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(convolution_34, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_73, (256, ), (1, ))
    assert_size_stride(mul_202, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_35, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_76, (256, ), (1, ))
    assert_size_stride(add_137, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(mean_5, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(relu_5, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(convolution_37, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(mul_211, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_38, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(squeeze_79, (1024, ), (1, ))
    assert_size_stride(mul_219, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(convolution_39, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_82, (256, ), (1, ))
    assert_size_stride(mul_227, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(view_31, (8192, 64), (64, 1))
    assert_size_stride(view_37, (8192, 64), (64, 1))
    assert_size_stride(bmm_3, (32, 256, 64), (16384, 64, 1))
    assert_size_stride(squeeze_85, (256, ), (1, ))
    assert_size_stride(mul_236, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_41, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(squeeze_88, (1024, ), (1, ))
    assert_size_stride(mul_244, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(convolution_42, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(squeeze_91, (512, ), (1, ))
    assert_size_stride(mul_252, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(view_55, (8192, 128), (128, 1))
    assert_size_stride(view_61, (8192, 128), (128, 1))
    assert_size_stride(view_71, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(avg_pool2d, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_94, (512, ), (1, ))
    assert_size_stride(mul_261, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_44, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
    assert_size_stride(squeeze_97, (1536, ), (1, ))
    assert_size_stride(convolution_45, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
    assert_size_stride(squeeze_100, (1536, ), (1, ))
    assert_size_stride(mul_276, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
    assert_size_stride(convolution_46, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_103, (512, ), (1, ))
    assert_size_stride(mul_284, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(view_79, (2048, 128), (128, 1))
    assert_size_stride(view_85, (2048, 128), (128, 1))
    assert_size_stride(bmm_7, (32, 64, 128), (8192, 128, 1))
    assert_size_stride(squeeze_106, (512, ), (1, ))
    assert_size_stride(mul_293, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_48, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
    assert_size_stride(squeeze_109, (1536, ), (1, ))
    assert_size_stride(mul_301, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
    assert_size_stride(convolution_49, (8, 1280, 8, 8), (81920, 1, 10240, 1280))
    assert_size_stride(squeeze_112, (1280, ), (1, ))
    assert_size_stride(clone_62, (8, 1280), (1280, 1))
    assert_size_stride(permute_33, (1000, 1280), (1280, 1))
    assert_size_stride(mul_311, (8, 1280, 8, 8), (81920, 1, 10240, 1280))
    assert_size_stride(unsqueeze_154, (1, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(mul_323, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
    assert_size_stride(unsqueeze_166, (1, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_335, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_178, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(permute_41, (32, 64, 64), (4096, 1, 64))
    assert_size_stride(permute_42, (32, 128, 64), (8192, 64, 1))
    assert_size_stride(alias_16, (32, 64, 64), (4096, 64, 1))
    assert_size_stride(permute_46, (15, 128), (128, 1))
    assert_size_stride(permute_52, (15, 128), (128, 1))
    assert_size_stride(permute_54, (32, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_55, (32, 64, 128), (8192, 1, 64))
    assert_size_stride(mul_350, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_190, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_362, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
    assert_size_stride(unsqueeze_202, (1, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(unsqueeze_214, (1, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_383, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_226, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(permute_62, (32, 256, 256), (65536, 1, 256))
    assert_size_stride(permute_63, (32, 128, 256), (32768, 256, 1))
    assert_size_stride(alias_17, (32, 256, 256), (65536, 256, 1))
    assert_size_stride(permute_67, (31, 128), (128, 1))
    assert_size_stride(permute_73, (31, 128), (128, 1))
    assert_size_stride(permute_75, (32, 128, 256), (32768, 256, 1))
    assert_size_stride(permute_76, (32, 256, 128), (32768, 1, 256))
    assert_size_stride(mul_398, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(unsqueeze_238, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_410, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(unsqueeze_250, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(mul_422, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_262, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(permute_83, (32, 256, 256), (65536, 1, 256))
    assert_size_stride(permute_84, (32, 64, 256), (16384, 256, 1))
    assert_size_stride(alias_18, (32, 256, 256), (65536, 256, 1))
    assert_size_stride(permute_88, (31, 64), (64, 1))
    assert_size_stride(permute_94, (31, 64), (64, 1))
    assert_size_stride(permute_96, (32, 64, 256), (16384, 256, 1))
    assert_size_stride(permute_97, (32, 256, 64), (16384, 1, 256))
    assert_size_stride(mul_437, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_274, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_449, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(unsqueeze_286, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_298, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_477, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_310, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_489, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(unsqueeze_322, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_334, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_346, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_526, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(unsqueeze_358, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_538, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(unsqueeze_370, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_550, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_382, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(permute_110, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_111, (32, 32, 1024), (32768, 1024, 1))
    assert_size_stride(alias_27, (32, 1024, 1024), (1048576, 1024, 1))
    assert_size_stride(permute_115, (63, 32), (32, 1))
    assert_size_stride(permute_121, (63, 32), (32, 1))
    assert_size_stride(permute_123, (32, 32, 1024), (32768, 1024, 1))
    assert_size_stride(permute_124, (32, 1024, 32), (32768, 1, 1024))
    assert_size_stride(mul_565, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_394, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_577, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(unsqueeze_406, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_418, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_605, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_430, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_617, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(unsqueeze_442, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_454, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_466, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_654, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(unsqueeze_478, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_666, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(unsqueeze_490, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_502, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_694, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(unsqueeze_514, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_706, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(unsqueeze_526, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_538, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_550, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_743, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(unsqueeze_562, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_755, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(unsqueeze_574, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_767, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(unsqueeze_586, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(mul_779, (8, 24, 128, 128), (393216, 1, 3072, 24))
    assert_size_stride(unsqueeze_598, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_33, out=buf0)
    del permute_33
    buf1 = empty((1000, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_62, out=buf1)
    del clone_62
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf4 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf5 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf6 = empty((8, 1280, 8, 8), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_mul_native_batch_norm_backward_sum_0(c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(mul_311.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(unsqueeze_154.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del buf0
    del buf4
    del convolution_49
    del mul_311
    del primals_83
    del squeeze_112
    del tangents_1
    del unsqueeze_154
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
    buf7 = aten.convolution_backward(buf6, mul_301, primals_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf6
    del mul_301
    del primals_146
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf10 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf11 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf12 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_1(c_void_p(buf8.data_ptr()), c_void_p(mul_323.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(unsqueeze_166.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    del convolution_48
    del primals_81
    del squeeze_109
    del unsqueeze_166
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf14 = aten.convolution_backward(buf13, mul_293, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_293
    del primals_145
    buf15 = buf14[0]
    buf16 = buf14[1]
    del buf14
    buf17 = empty((512, ), device='cpu', dtype=torch.float32)
    buf18 = empty((512, ), device='cpu', dtype=torch.float32)
    buf19 = empty((512, ), device='cpu', dtype=torch.float32)
    buf20 = empty((8, 512, 8, 8), device='cpu', dtype=torch.float32)
    cpp_fused_mul_native_batch_norm_backward_2(c_void_p(buf15.data_ptr()), c_void_p(mul_335.data_ptr()), c_void_p(bmm_7.data_ptr()), c_void_p(unsqueeze_178.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()))
    del bmm_7
    del mul_335
    del primals_79
    del squeeze_106
    del unsqueeze_178
    buf21 = reinterpret_tensor(buf15, (32, 64, 128), (8192, 128, 1), 0); del buf15  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_41, reinterpret_tensor(buf20, (32, 64, 128), (8192, 1, 64), 0), out=buf21)
    del permute_41
    buf22 = empty((32, 64, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf20, (32, 64, 128), (8192, 1, 64), 0), permute_42, out=buf22)
    del permute_42
    buf23 = empty_strided((32, 64, 1), (64, 1, 2048), device='cpu', dtype=torch.float32)
    buf24 = empty((32, 8, 1, 8, 8), device='cpu', dtype=torch.float32)
    buf25 = empty((2048, 15), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_sum_view_3(c_void_p(buf22.data_ptr()), c_void_p(alias_16.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()))
    buf26 = empty((15, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf25, (15, 2048), (1, 15), 0), view_85, out=buf26)
    del view_85
    buf27 = reinterpret_tensor(buf20, (2048, 128), (128, 1), 0); del buf20  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf25, permute_46, out=buf27)
    del permute_46
    buf28 = buf24; del buf24  # reuse
    buf29 = buf25; del buf25  # reuse
    cpp_fused_sum_view_4(c_void_p(buf22.data_ptr()), c_void_p(alias_16.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()))
    del buf28
    buf30 = empty((15, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf29, (15, 2048), (1, 15), 0), view_79, out=buf30)
    del view_79
    buf31 = empty((2048, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf29, permute_52, out=buf31)
    del buf29
    del permute_52
    buf32 = buf22; del buf22  # reuse
    cpp_fused__softmax_backward_data_mul_5(c_void_p(buf32.data_ptr()), c_void_p(alias_16.data_ptr()), c_void_p(buf23.data_ptr()))
    del alias_16
    buf33 = empty((32, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_54, buf32, out=buf33)
    del permute_54
    buf34 = empty((32, 64, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf32, permute_55, out=buf34)
    del permute_55
    buf35 = reinterpret_tensor(buf13, (8, 1536, 8, 8), (98304, 64, 8, 1), 0); del buf13  # reuse
    cpp_fused_cat_convolution_backward_6(c_void_p(buf27.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf35.data_ptr()))
    del buf21
    del buf27
    del buf31
    del buf33
    del buf34
    # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]
    buf36 = aten.convolution_backward(buf35, mul_284, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_284
    del primals_144
    buf37 = buf36[0]
    buf38 = buf36[1]
    del buf36
    buf39 = buf18; del buf18  # reuse
    buf40 = empty((512, ), device='cpu', dtype=torch.float32)
    buf41 = empty((512, ), device='cpu', dtype=torch.float32)
    buf42 = buf37; del buf37  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_7(c_void_p(buf42.data_ptr()), c_void_p(mul_350.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(unsqueeze_190.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()))
    del convolution_46
    del mul_350
    del primals_75
    del squeeze_103
    del unsqueeze_190
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf43 = aten.convolution_backward(buf42, mul_276, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf42
    del mul_276
    del primals_143
    buf44 = buf43[0]
    buf45 = buf43[1]
    del buf43
    buf46 = buf11; del buf11  # reuse
    buf47 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf54 = empty((1536, ), device='cpu', dtype=torch.float32)
    buf48 = reinterpret_tensor(buf35, (8, 1536, 8, 8), (98304, 1, 12288, 1536), 0); del buf35  # reuse
    buf55 = empty_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cpu', dtype=torch.float32)
    buf50 = buf48; del buf48  # reuse
    buf57 = buf55; del buf55  # reuse
    buf49 = buf47; del buf47  # reuse
    cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_8(c_void_p(buf50.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(mul_323.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(mul_362.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(unsqueeze_202.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(unsqueeze_214.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf54.data_ptr()))
    del buf44
    del buf8
    del convolution_44
    del convolution_45
    del mul_323
    del mul_362
    del primals_71
    del primals_73
    del squeeze_100
    del unsqueeze_202
    del unsqueeze_214
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf51 = aten.convolution_backward(buf50, mul_244, primals_142, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf50
    del primals_142
    buf52 = buf51[0]
    buf53 = buf51[1]
    del buf51
    buf56 = buf54; del buf54  # reuse
    cpp_fused_native_batch_norm_backward_9(c_void_p(buf56.data_ptr()), c_void_p(squeeze_97.data_ptr()))
    del squeeze_97
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf58 = aten.convolution_backward(buf57, mul_261, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf57
    del mul_261
    del primals_141
    buf59 = buf58[0]
    buf60 = buf58[1]
    del buf58
    buf61 = buf40; del buf40  # reuse
    buf62 = empty((512, ), device='cpu', dtype=torch.float32)
    buf63 = empty((512, ), device='cpu', dtype=torch.float32)
    buf64 = buf59; del buf59  # reuse
    buf65 = empty((8, 512, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused_avg_pool2d_backward_mul_native_batch_norm_backward_10(c_void_p(buf64.data_ptr()), c_void_p(mul_383.data_ptr()), c_void_p(avg_pool2d.data_ptr()), c_void_p(unsqueeze_226.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf65.data_ptr()))
    del avg_pool2d
    del buf64
    del mul_383
    del primals_69
    del squeeze_94
    del unsqueeze_226
    buf66 = empty((32, 256, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_62, reinterpret_tensor(buf65, (32, 256, 128), (32768, 1, 256), 0), out=buf66)
    del permute_62
    buf67 = empty((32, 256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf65, (32, 256, 128), (32768, 1, 256), 0), permute_63, out=buf67)
    del permute_63
    buf68 = empty_strided((32, 256, 1), (256, 1, 8192), device='cpu', dtype=torch.float32)
    buf69 = reinterpret_tensor(buf32, (32, 16, 1, 16, 16), (4096, 256, 256, 16, 1), 0); del buf32  # reuse
    buf70 = empty((8192, 31), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_sum_view_11(c_void_p(buf67.data_ptr()), c_void_p(alias_17.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    buf71 = empty((31, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf70, (31, 8192), (1, 31), 0), view_61, out=buf71)
    del view_61
    buf72 = reinterpret_tensor(buf65, (8192, 128), (128, 1), 0); del buf65  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf70, permute_67, out=buf72)
    del permute_67
    buf73 = buf69; del buf69  # reuse
    buf74 = buf70; del buf70  # reuse
    cpp_fused_sum_view_12(c_void_p(buf67.data_ptr()), c_void_p(alias_17.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()))
    buf75 = empty((31, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf74, (31, 8192), (1, 31), 0), view_55, out=buf75)
    del view_55
    buf76 = empty((8192, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf74, permute_73, out=buf76)
    del permute_73
    buf77 = buf67; del buf67  # reuse
    cpp_fused__softmax_backward_data_mul_13(c_void_p(buf77.data_ptr()), c_void_p(alias_17.data_ptr()), c_void_p(buf68.data_ptr()))
    del alias_17
    buf78 = empty((32, 128, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_75, buf77, out=buf78)
    del permute_75
    buf79 = empty((32, 256, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf77, permute_76, out=buf79)
    del permute_76
    buf80 = empty((8, 1536, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused_cat_convolution_backward_14(c_void_p(buf72.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf80.data_ptr()))
    del buf66
    del buf72
    # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]
    buf81 = aten.convolution_backward(buf80, mul_252, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_252
    del primals_140
    buf82 = buf81[0]
    buf83 = buf81[1]
    del buf81
    buf84 = buf62; del buf62  # reuse
    buf85 = empty((512, ), device='cpu', dtype=torch.float32)
    buf86 = empty((512, ), device='cpu', dtype=torch.float32)
    buf87 = buf82; del buf82  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_15(c_void_p(buf87.data_ptr()), c_void_p(mul_398.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(unsqueeze_238.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()))
    del convolution_42
    del mul_398
    del primals_65
    del squeeze_91
    del unsqueeze_238
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf88 = aten.convolution_backward(buf87, mul_244, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_244
    del primals_139
    buf89 = buf88[0]
    buf90 = buf88[1]
    del buf88
    buf91 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf92 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf93 = reinterpret_tensor(buf77, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf77  # reuse
    buf94 = buf92; del buf92  # reuse
    cpp_fused_add_mul_native_batch_norm_backward_16(c_void_p(buf94.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(mul_410.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(unsqueeze_250.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf93.data_ptr()))
    del convolution_41
    del primals_63
    del squeeze_88
    del unsqueeze_250
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf95 = aten.convolution_backward(buf93, mul_236, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_236
    del primals_138
    buf96 = buf95[0]
    buf97 = buf95[1]
    del buf95
    buf98 = empty((256, ), device='cpu', dtype=torch.float32)
    buf99 = empty((256, ), device='cpu', dtype=torch.float32)
    buf100 = empty((256, ), device='cpu', dtype=torch.float32)
    buf101 = empty((8, 256, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused_mul_native_batch_norm_backward_17(c_void_p(buf96.data_ptr()), c_void_p(mul_422.data_ptr()), c_void_p(bmm_3.data_ptr()), c_void_p(unsqueeze_262.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()))
    del bmm_3
    del mul_422
    del primals_61
    del squeeze_85
    del unsqueeze_262
    buf102 = reinterpret_tensor(buf96, (32, 256, 64), (16384, 64, 1), 0); del buf96  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_83, reinterpret_tensor(buf101, (32, 256, 64), (16384, 1, 256), 0), out=buf102)
    del permute_83
    buf103 = reinterpret_tensor(buf93, (32, 256, 256), (65536, 256, 1), 0); del buf93  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf101, (32, 256, 64), (16384, 1, 256), 0), permute_84, out=buf103)
    del permute_84
    buf104 = buf68; del buf68  # reuse
    buf105 = buf73; del buf73  # reuse
    buf106 = buf74; del buf74  # reuse
    cpp_fused__softmax_backward_data_sum_view_18(c_void_p(buf103.data_ptr()), c_void_p(alias_18.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    buf107 = empty((31, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf106, (31, 8192), (1, 31), 0), view_37, out=buf107)
    del view_37
    buf108 = reinterpret_tensor(buf101, (8192, 64), (64, 1), 0); del buf101  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf106, permute_88, out=buf108)
    del permute_88
    buf109 = buf105; del buf105  # reuse
    buf110 = buf106; del buf106  # reuse
    cpp_fused_sum_view_19(c_void_p(buf103.data_ptr()), c_void_p(alias_18.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()))
    del buf109
    buf111 = empty((31, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf110, (31, 8192), (1, 31), 0), view_31, out=buf111)
    del view_31
    buf112 = empty((8192, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf110, permute_94, out=buf112)
    del buf110
    del permute_94
    buf113 = buf103; del buf103  # reuse
    cpp_fused__softmax_backward_data_mul_20(c_void_p(buf113.data_ptr()), c_void_p(alias_18.data_ptr()), c_void_p(buf104.data_ptr()))
    del alias_18
    del buf104
    buf114 = empty((32, 64, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_96, buf113, out=buf114)
    del permute_96
    buf115 = empty((32, 256, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf113, permute_97, out=buf115)
    del permute_97
    buf116 = empty((8, 768, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused_cat_convolution_backward_21(c_void_p(buf108.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf116.data_ptr()))
    del buf102
    del buf108
    del buf112
    del buf114
    del buf115
    # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]
    buf117 = aten.convolution_backward(buf116, mul_227, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf116
    del mul_227
    del primals_137
    buf118 = buf117[0]
    buf119 = buf117[1]
    del buf117
    buf120 = buf99; del buf99  # reuse
    buf121 = empty((256, ), device='cpu', dtype=torch.float32)
    buf122 = empty((256, ), device='cpu', dtype=torch.float32)
    buf123 = buf118; del buf118  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_22(c_void_p(buf123.data_ptr()), c_void_p(mul_437.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(unsqueeze_274.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()))
    del convolution_39
    del mul_437
    del primals_57
    del squeeze_82
    del unsqueeze_274
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf124 = aten.convolution_backward(buf123, mul_219, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf123
    del mul_219
    del primals_136
    buf125 = buf124[0]
    buf126 = buf124[1]
    del buf124
    buf127 = buf125; del buf125  # reuse
    buf128 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf129 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf130 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf131 = reinterpret_tensor(buf113, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf113  # reuse
    cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_23(c_void_p(buf127.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(mul_410.data_ptr()), c_void_p(mul_449.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(unsqueeze_286.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()))
    del buf52
    del convolution_38
    del mul_410
    del mul_449
    del primals_55
    del squeeze_79
    del unsqueeze_286
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf132 = aten.convolution_backward(buf131, mul_211, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_211
    del primals_135
    buf133 = buf132[0]
    buf134 = buf132[1]
    del buf132
    buf135 = reinterpret_tensor(buf23, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf23  # reuse
    buf136 = reinterpret_tensor(buf135, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf135  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_24(c_void_p(buf136.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(add_137.data_ptr()), c_void_p(convolution_37.data_ptr()))
    # Source Nodes: [sigmoid_5], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf137 = aten.convolution_backward(buf136, relu_5, primals_133, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf136
    del primals_133
    buf138 = buf137[0]
    buf139 = buf137[1]
    buf140 = buf137[2]
    del buf137
    buf141 = buf138; del buf138  # reuse
    cpp_fused_convolution_backward_threshold_backward_25(c_void_p(buf141.data_ptr()), c_void_p(relu_5.data_ptr()))
    del relu_5
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf142 = aten.convolution_backward(buf141, mean_5, primals_131, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_5
    del primals_131
    buf143 = buf142[0]
    buf144 = buf142[1]
    buf145 = buf142[2]
    del buf142
    buf146 = buf121; del buf121  # reuse
    buf147 = empty((256, ), device='cpu', dtype=torch.float32)
    buf148 = buf133; del buf133  # reuse
    buf149 = buf147; del buf147  # reuse
    buf150 = buf148; del buf148  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_26(c_void_p(buf150.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(add_137.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_298.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf146.data_ptr()))
    del add_137
    del convolution_35
    del convolution_37
    del primals_53
    del squeeze_76
    del unsqueeze_298
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf151 = aten.convolution_backward(buf150, mul_202, primals_130, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf150
    del mul_202
    del primals_130
    buf152 = buf151[0]
    buf153 = buf151[1]
    del buf151
    buf154 = empty((256, ), device='cpu', dtype=torch.float32)
    buf155 = empty((256, ), device='cpu', dtype=torch.float32)
    buf156 = empty((256, ), device='cpu', dtype=torch.float32)
    buf157 = buf152; del buf152  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_27(c_void_p(buf157.data_ptr()), c_void_p(mul_477.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_310.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()))
    del convolution_34
    del mul_477
    del primals_51
    del squeeze_73
    del unsqueeze_310
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf158 = aten.convolution_backward(buf157, mul_194, primals_129, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf157
    del mul_194
    del primals_129
    buf159 = buf158[0]
    buf160 = buf158[1]
    del buf158
    buf161 = buf129; del buf129  # reuse
    buf162 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf168 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf163 = buf131; del buf131  # reuse
    buf169 = buf89; del buf89  # reuse
    buf164 = buf162; del buf162  # reuse
    cpp_fused_add_mul_native_batch_norm_backward_28(c_void_p(buf164.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(mul_489.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_322.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(unsqueeze_334.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf169.data_ptr()))
    del buf127
    del buf159
    del convolution_32
    del convolution_33
    del mul_489
    del primals_47
    del primals_49
    del squeeze_70
    del unsqueeze_322
    del unsqueeze_334
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf165 = aten.convolution_backward(buf163, mul_162, primals_128, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf163
    del primals_128
    buf166 = buf165[0]
    buf167 = buf165[1]
    del buf165
    buf170 = buf168; del buf168  # reuse
    cpp_fused_native_batch_norm_backward_29(c_void_p(buf170.data_ptr()), c_void_p(squeeze_67.data_ptr()))
    del squeeze_67
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf171 = aten.convolution_backward(buf169, mul_179, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf169
    del mul_179
    del primals_127
    buf172 = buf171[0]
    buf173 = buf171[1]
    del buf171
    buf174 = reinterpret_tensor(buf143, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf143  # reuse
    buf175 = reinterpret_tensor(buf174, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf174  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_30(c_void_p(buf175.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(add_116.data_ptr()), c_void_p(convolution_31.data_ptr()))
    # Source Nodes: [sigmoid_4], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf176 = aten.convolution_backward(buf175, relu_4, primals_125, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf175
    del primals_125
    buf177 = buf176[0]
    buf178 = buf176[1]
    buf179 = buf176[2]
    del buf176
    buf180 = buf177; del buf177  # reuse
    cpp_fused_convolution_backward_threshold_backward_31(c_void_p(buf180.data_ptr()), c_void_p(relu_4.data_ptr()))
    del relu_4
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf181 = aten.convolution_backward(buf180, mean_4, primals_123, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_4
    del primals_123
    buf182 = buf181[0]
    buf183 = buf181[1]
    buf184 = buf181[2]
    del buf181
    buf185 = buf155; del buf155  # reuse
    buf186 = empty((256, ), device='cpu', dtype=torch.float32)
    buf187 = buf172; del buf172  # reuse
    buf188 = buf186; del buf186  # reuse
    buf189 = buf187; del buf187  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_32(c_void_p(buf189.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(add_116.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_346.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf185.data_ptr()))
    del add_116
    del buf182
    del convolution_29
    del convolution_31
    del primals_45
    del squeeze_64
    del unsqueeze_346
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf190 = aten.convolution_backward(buf189, mul_170, primals_122, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf189
    del mul_170
    del primals_122
    buf191 = buf190[0]
    buf192 = buf190[1]
    del buf190
    buf193 = empty((256, ), device='cpu', dtype=torch.float32)
    buf194 = empty((256, ), device='cpu', dtype=torch.float32)
    buf195 = empty((256, ), device='cpu', dtype=torch.float32)
    buf196 = buf191; del buf191  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_33(c_void_p(buf196.data_ptr()), c_void_p(mul_526.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(unsqueeze_358.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()))
    del convolution_28
    del mul_526
    del primals_43
    del squeeze_61
    del unsqueeze_358
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf197 = aten.convolution_backward(buf196, mul_162, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf196
    del mul_162
    del primals_121
    buf198 = buf197[0]
    buf199 = buf197[1]
    del buf197
    buf200 = buf85; del buf85  # reuse
    buf201 = empty((512, ), device='cpu', dtype=torch.float32)
    buf202 = empty_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    buf203 = buf201; del buf201  # reuse
    cpp_fused_add_mul_native_batch_norm_backward_34(c_void_p(buf203.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(mul_538.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(unsqueeze_370.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf202.data_ptr()))
    del convolution_27
    del primals_41
    del squeeze_58
    del unsqueeze_370
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf204 = aten.convolution_backward(buf202, mul_154, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_154
    del primals_120
    buf205 = buf204[0]
    buf206 = buf204[1]
    del buf204
    buf207 = reinterpret_tensor(buf180, (128, ), (1, ), 0); del buf180  # reuse
    buf208 = reinterpret_tensor(buf141, (128, ), (1, ), 0); del buf141  # reuse
    buf209 = empty((128, ), device='cpu', dtype=torch.float32)
    buf210 = reinterpret_tensor(buf87, (8, 128, 32, 32), (131072, 1024, 32, 1), 0); del buf87  # reuse
    cpp_fused_mul_native_batch_norm_backward_35(c_void_p(buf205.data_ptr()), c_void_p(mul_550.data_ptr()), c_void_p(bmm_1.data_ptr()), c_void_p(unsqueeze_382.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()))
    del bmm_1
    del mul_550
    del primals_39
    del squeeze_55
    del unsqueeze_382
    buf211 = reinterpret_tensor(buf205, (32, 1024, 32), (32768, 32, 1), 0); del buf205  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_110, reinterpret_tensor(buf210, (32, 1024, 32), (32768, 1, 1024), 0), out=buf211)
    del permute_110
    buf212 = empty((32, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf210, (32, 1024, 32), (32768, 1, 1024), 0), permute_111, out=buf212)
    del permute_111
    buf213 = empty_strided((32, 1024, 1), (1024, 1, 32768), device='cpu', dtype=torch.float32)
    buf214 = reinterpret_tensor(buf210, (32, 32, 1, 32, 32), (32768, 1024, 1024, 32, 1), 0); del buf210  # reuse
    buf215 = empty((32768, 63), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_sum_view_36(c_void_p(buf212.data_ptr()), c_void_p(alias_27.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()))
    buf216 = empty((63, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf215, (63, 32768), (1, 63), 0), view_13, out=buf216)
    del view_13
    buf217 = reinterpret_tensor(buf214, (32768, 32), (32, 1), 0); del buf214  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf215, permute_115, out=buf217)
    del permute_115
    buf218 = reinterpret_tensor(buf79, (32, 32, 1, 32, 32), (32768, 1024, 1024, 32, 1), 0); del buf79  # reuse
    buf219 = buf215; del buf215  # reuse
    cpp_fused_sum_view_37(c_void_p(buf212.data_ptr()), c_void_p(alias_27.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()))
    buf220 = empty((63, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf219, (63, 32768), (1, 63), 0), view_7, out=buf220)
    del view_7
    buf221 = reinterpret_tensor(buf218, (32768, 32), (32, 1), 0); del buf218  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf219, permute_121, out=buf221)
    del buf219
    del permute_121
    buf222 = buf212; del buf212  # reuse
    cpp_fused__softmax_backward_data_mul_38(c_void_p(buf222.data_ptr()), c_void_p(alias_27.data_ptr()), c_void_p(buf213.data_ptr()))
    del alias_27
    del buf213
    buf223 = reinterpret_tensor(buf78, (32, 32, 1024), (32768, 1024, 1), 0); del buf78  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_123, buf222, out=buf223)
    del permute_123
    buf224 = reinterpret_tensor(buf76, (32, 1024, 32), (32768, 32, 1), 0); del buf76  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf222, permute_124, out=buf224)
    del buf222
    del permute_124
    buf225 = reinterpret_tensor(buf80, (8, 384, 32, 32), (393216, 1024, 32, 1), 0); del buf80  # reuse
    cpp_fused_cat_convolution_backward_39(c_void_p(buf217.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf225.data_ptr()))
    del buf211
    del buf217
    del buf221
    del buf223
    del buf224
    # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]
    buf226 = aten.convolution_backward(buf225, mul_145, primals_119, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf225
    del mul_145
    del primals_119
    buf227 = buf226[0]
    buf228 = buf226[1]
    del buf226
    buf229 = buf208; del buf208  # reuse
    buf230 = empty((128, ), device='cpu', dtype=torch.float32)
    buf231 = empty((128, ), device='cpu', dtype=torch.float32)
    buf232 = buf227; del buf227  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_40(c_void_p(buf232.data_ptr()), c_void_p(mul_565.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_394.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()))
    del convolution_25
    del mul_565
    del primals_35
    del squeeze_52
    del unsqueeze_394
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf233 = aten.convolution_backward(buf232, mul_137, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf232
    del mul_137
    del primals_118
    buf234 = buf233[0]
    buf235 = buf233[1]
    del buf233
    buf236 = buf166; del buf166  # reuse
    buf237 = empty((512, ), device='cpu', dtype=torch.float32)
    buf238 = empty((512, ), device='cpu', dtype=torch.float32)
    buf239 = empty((512, ), device='cpu', dtype=torch.float32)
    buf240 = buf202; del buf202  # reuse
    cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_41(c_void_p(buf236.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(mul_538.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(mul_577.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_406.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()))
    del buf198
    del convolution_24
    del mul_538
    del mul_577
    del primals_33
    del squeeze_49
    del unsqueeze_406
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf241 = aten.convolution_backward(buf240, mul_129, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_129
    del primals_117
    buf242 = buf241[0]
    buf243 = buf241[1]
    del buf241
    buf244 = empty_strided((8, 128, 1, 1), (128, 1, 1024, 1024), device='cpu', dtype=torch.float32)
    buf245 = reinterpret_tensor(buf244, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf244  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_42(c_void_p(buf245.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(add_82.data_ptr()), c_void_p(convolution_23.data_ptr()))
    # Source Nodes: [sigmoid_3], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf246 = aten.convolution_backward(buf245, relu_3, primals_115, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf245
    del primals_115
    buf247 = buf246[0]
    buf248 = buf246[1]
    buf249 = buf246[2]
    del buf246
    buf250 = buf247; del buf247  # reuse
    cpp_fused_convolution_backward_threshold_backward_43(c_void_p(buf250.data_ptr()), c_void_p(relu_3.data_ptr()))
    del relu_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf251 = aten.convolution_backward(buf250, mean_3, primals_113, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_3
    del primals_113
    buf252 = buf251[0]
    buf253 = buf251[1]
    buf254 = buf251[2]
    del buf251
    buf255 = buf230; del buf230  # reuse
    buf256 = empty((128, ), device='cpu', dtype=torch.float32)
    buf257 = buf242; del buf242  # reuse
    buf258 = buf256; del buf256  # reuse
    buf259 = buf257; del buf257  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_44(c_void_p(buf259.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(add_82.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_418.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf255.data_ptr()))
    del add_82
    del convolution_21
    del convolution_23
    del primals_31
    del squeeze_46
    del unsqueeze_418
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf260 = aten.convolution_backward(buf259, mul_120, primals_112, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf259
    del mul_120
    del primals_112
    buf261 = buf260[0]
    buf262 = buf260[1]
    del buf260
    buf263 = empty((128, ), device='cpu', dtype=torch.float32)
    buf264 = empty((128, ), device='cpu', dtype=torch.float32)
    buf265 = empty((128, ), device='cpu', dtype=torch.float32)
    buf266 = buf261; del buf261  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_45(c_void_p(buf266.data_ptr()), c_void_p(mul_605.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_430.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()))
    del convolution_20
    del mul_605
    del primals_29
    del squeeze_43
    del unsqueeze_430
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf267 = aten.convolution_backward(buf266, mul_112, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf266
    del mul_112
    del primals_111
    buf268 = buf267[0]
    buf269 = buf267[1]
    del buf267
    buf270 = buf238; del buf238  # reuse
    buf271 = empty((512, ), device='cpu', dtype=torch.float32)
    buf277 = empty((512, ), device='cpu', dtype=torch.float32)
    buf272 = buf240; del buf240  # reuse
    buf278 = buf234; del buf234  # reuse
    buf273 = buf271; del buf271  # reuse
    cpp_fused_add_mul_native_batch_norm_backward_46(c_void_p(buf273.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(mul_617.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_442.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_454.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf278.data_ptr()))
    del buf236
    del buf268
    del convolution_18
    del convolution_19
    del mul_617
    del primals_25
    del primals_27
    del squeeze_40
    del unsqueeze_442
    del unsqueeze_454
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf274 = aten.convolution_backward(buf272, mul_80, primals_110, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf272
    del primals_110
    buf275 = buf274[0]
    buf276 = buf274[1]
    del buf274
    buf279 = buf277; del buf277  # reuse
    cpp_fused_native_batch_norm_backward_47(c_void_p(buf279.data_ptr()), c_void_p(squeeze_37.data_ptr()))
    del squeeze_37
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf280 = aten.convolution_backward(buf278, mul_97, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf278
    del mul_97
    del primals_109
    buf281 = buf280[0]
    buf282 = buf280[1]
    del buf280
    buf283 = reinterpret_tensor(buf252, (8, 128, 1, 1), (128, 1, 1024, 1024), 0); del buf252  # reuse
    buf284 = reinterpret_tensor(buf283, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf283  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_48(c_void_p(buf284.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(add_61.data_ptr()), c_void_p(convolution_17.data_ptr()))
    # Source Nodes: [sigmoid_2], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf285 = aten.convolution_backward(buf284, relu_2, primals_107, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf284
    del primals_107
    buf286 = buf285[0]
    buf287 = buf285[1]
    buf288 = buf285[2]
    del buf285
    buf289 = buf286; del buf286  # reuse
    cpp_fused_convolution_backward_threshold_backward_49(c_void_p(buf289.data_ptr()), c_void_p(relu_2.data_ptr()))
    del relu_2
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf290 = aten.convolution_backward(buf289, mean_2, primals_105, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_2
    del primals_105
    buf291 = buf290[0]
    buf292 = buf290[1]
    buf293 = buf290[2]
    del buf290
    buf294 = buf264; del buf264  # reuse
    buf295 = empty((128, ), device='cpu', dtype=torch.float32)
    buf296 = buf281; del buf281  # reuse
    buf297 = buf295; del buf295  # reuse
    buf298 = buf296; del buf296  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_50(c_void_p(buf298.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(add_61.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_466.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf294.data_ptr()))
    del add_61
    del buf291
    del convolution_15
    del convolution_17
    del primals_23
    del squeeze_34
    del unsqueeze_466
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf299 = aten.convolution_backward(buf298, mul_88, primals_104, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf298
    del mul_88
    del primals_104
    buf300 = buf299[0]
    buf301 = buf299[1]
    del buf299
    buf302 = empty((128, ), device='cpu', dtype=torch.float32)
    buf303 = empty((128, ), device='cpu', dtype=torch.float32)
    buf304 = empty((128, ), device='cpu', dtype=torch.float32)
    buf305 = buf300; del buf300  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_51(c_void_p(buf305.data_ptr()), c_void_p(mul_654.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_478.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()))
    del buf303
    del convolution_14
    del mul_654
    del primals_21
    del squeeze_31
    del unsqueeze_478
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf306 = aten.convolution_backward(buf305, mul_80, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf305
    del mul_80
    del primals_103
    buf307 = buf306[0]
    buf308 = buf306[1]
    del buf306
    buf309 = buf194; del buf194  # reuse
    buf310 = empty((256, ), device='cpu', dtype=torch.float32)
    buf311 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    buf312 = buf310; del buf310  # reuse
    cpp_fused_add_mul_native_batch_norm_backward_52(c_void_p(buf312.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(mul_666.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_490.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf311.data_ptr()))
    del convolution_13
    del primals_19
    del squeeze_28
    del unsqueeze_490
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf313 = aten.convolution_backward(buf311, mul_72, primals_102, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_72
    del primals_102
    buf314 = buf313[0]
    buf315 = buf313[1]
    del buf313
    buf316 = empty_strided((8, 64, 1, 1), (64, 1, 512, 512), device='cpu', dtype=torch.float32)
    buf317 = reinterpret_tensor(buf316, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf316  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_53(c_void_p(buf317.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(add_45.data_ptr()), c_void_p(convolution_12.data_ptr()))
    # Source Nodes: [sigmoid_1], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf318 = aten.convolution_backward(buf317, relu_1, primals_100, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf317
    del primals_100
    buf319 = buf318[0]
    buf320 = buf318[1]
    buf321 = buf318[2]
    del buf318
    buf322 = buf319; del buf319  # reuse
    cpp_fused_convolution_backward_threshold_backward_54(c_void_p(buf322.data_ptr()), c_void_p(relu_1.data_ptr()))
    del relu_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf323 = aten.convolution_backward(buf322, mean_1, primals_98, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_1
    del primals_98
    buf324 = buf323[0]
    buf325 = buf323[1]
    buf326 = buf323[2]
    del buf323
    buf327 = reinterpret_tensor(buf322, (64, ), (1, ), 0); del buf322  # reuse
    buf328 = reinterpret_tensor(buf289, (64, ), (1, ), 0); del buf289  # reuse
    buf329 = buf314; del buf314  # reuse
    buf330 = buf328; del buf328  # reuse
    buf331 = buf329; del buf329  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_55(c_void_p(buf331.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(add_45.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_502.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf327.data_ptr()))
    del add_45
    del convolution_10
    del convolution_12
    del primals_17
    del squeeze_25
    del unsqueeze_502
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf332 = aten.convolution_backward(buf331, mul_63, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf331
    del mul_63
    del primals_97
    buf333 = buf332[0]
    buf334 = buf332[1]
    del buf332
    buf335 = reinterpret_tensor(buf250, (64, ), (1, ), 0); del buf250  # reuse
    buf336 = empty((64, ), device='cpu', dtype=torch.float32)
    buf337 = empty((64, ), device='cpu', dtype=torch.float32)
    buf338 = buf333; del buf333  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_56(c_void_p(buf338.data_ptr()), c_void_p(mul_694.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_514.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()))
    del convolution_9
    del mul_694
    del primals_15
    del squeeze_22
    del unsqueeze_514
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf339 = aten.convolution_backward(buf338, mul_55, primals_96, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf338
    del mul_55
    del primals_96
    buf340 = buf339[0]
    buf341 = buf339[1]
    del buf339
    buf342 = buf275; del buf275  # reuse
    buf343 = empty((256, ), device='cpu', dtype=torch.float32)
    buf344 = empty((256, ), device='cpu', dtype=torch.float32)
    buf350 = empty((256, ), device='cpu', dtype=torch.float32)
    buf345 = empty((256, ), device='cpu', dtype=torch.float32)
    buf346 = buf311; del buf311  # reuse
    buf352 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_57(c_void_p(buf342.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(mul_666.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(mul_706.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_526.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_538.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf352.data_ptr()))
    del buf307
    del buf340
    del buf342
    del buf344
    del convolution_7
    del convolution_8
    del mul_666
    del mul_706
    del primals_11
    del primals_13
    del squeeze_19
    del unsqueeze_526
    del unsqueeze_538
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf347 = aten.convolution_backward(buf346, mul_23, primals_95, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf346
    del primals_95
    buf348 = buf347[0]
    buf349 = buf347[1]
    del buf347
    buf351 = buf350; del buf350  # reuse
    cpp_fused_native_batch_norm_backward_58(c_void_p(buf351.data_ptr()), c_void_p(squeeze_16.data_ptr()))
    del squeeze_16
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf353 = aten.convolution_backward(buf352, mul_40, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf352
    del mul_40
    del primals_94
    buf354 = buf353[0]
    buf355 = buf353[1]
    del buf353
    buf356 = reinterpret_tensor(buf324, (8, 64, 1, 1), (64, 1, 512, 512), 0); del buf324  # reuse
    buf357 = reinterpret_tensor(buf356, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf356  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_59(c_void_p(buf357.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(add_24.data_ptr()), c_void_p(convolution_6.data_ptr()))
    # Source Nodes: [sigmoid], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf358 = aten.convolution_backward(buf357, relu, primals_92, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf357
    del primals_92
    buf359 = buf358[0]
    buf360 = buf358[1]
    buf361 = buf358[2]
    del buf358
    buf362 = buf359; del buf359  # reuse
    cpp_fused_convolution_backward_threshold_backward_60(c_void_p(buf362.data_ptr()), c_void_p(relu.data_ptr()))
    del relu
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf363 = aten.convolution_backward(buf362, mean, primals_90, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean
    del primals_90
    buf364 = buf363[0]
    buf365 = buf363[1]
    buf366 = buf363[2]
    del buf363
    buf367 = reinterpret_tensor(buf362, (64, ), (1, ), 0); del buf362  # reuse
    buf368 = buf336; del buf336  # reuse
    buf369 = buf354; del buf354  # reuse
    buf370 = buf368; del buf368  # reuse
    buf371 = buf369; del buf369  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_61(c_void_p(buf371.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(add_24.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_550.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf367.data_ptr()))
    del add_24
    del buf364
    del convolution_4
    del convolution_6
    del primals_9
    del squeeze_13
    del unsqueeze_550
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf372 = aten.convolution_backward(buf371, mul_31, primals_89, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf371
    del mul_31
    del primals_89
    buf373 = buf372[0]
    buf374 = buf372[1]
    del buf372
    buf375 = empty((64, ), device='cpu', dtype=torch.float32)
    buf376 = empty((64, ), device='cpu', dtype=torch.float32)
    buf377 = empty((64, ), device='cpu', dtype=torch.float32)
    buf378 = buf373; del buf373  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_62(c_void_p(buf378.data_ptr()), c_void_p(mul_743.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_562.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()))
    del convolution_3
    del mul_743
    del primals_7
    del squeeze_10
    del unsqueeze_562
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf379 = aten.convolution_backward(buf378, mul_23, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf378
    del mul_23
    del primals_88
    buf380 = buf379[0]
    buf381 = buf379[1]
    del buf379
    buf382 = buf376; del buf376  # reuse
    buf383 = empty((64, ), device='cpu', dtype=torch.float32)
    buf384 = buf348; del buf348  # reuse
    buf385 = buf383; del buf383  # reuse
    cpp_fused_add_mul_native_batch_norm_backward_63(c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(mul_755.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_574.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf382.data_ptr()))
    del buf380
    del convolution_2
    del mul_755
    del primals_5
    del squeeze_7
    del unsqueeze_574
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf386 = aten.convolution_backward(buf384, mul_15, primals_87, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf384
    del mul_15
    del primals_87
    buf387 = buf386[0]
    buf388 = buf386[1]
    del buf386
    buf389 = empty((32, ), device='cpu', dtype=torch.float32)
    buf390 = empty((32, ), device='cpu', dtype=torch.float32)
    buf391 = empty((32, ), device='cpu', dtype=torch.float32)
    buf392 = buf387; del buf387  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_64(c_void_p(buf392.data_ptr()), c_void_p(mul_767.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_586.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf391.data_ptr()))
    del buf390
    del convolution_1
    del mul_767
    del primals_3
    del squeeze_4
    del unsqueeze_586
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf393 = aten.convolution_backward(buf392, mul_7, primals_86, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf392
    del mul_7
    del primals_86
    buf394 = buf393[0]
    buf395 = buf393[1]
    del buf393
    buf396 = empty((24, ), device='cpu', dtype=torch.float32)
    buf397 = empty((24, ), device='cpu', dtype=torch.float32)
    buf398 = empty((24, ), device='cpu', dtype=torch.float32)
    buf399 = buf394; del buf394  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_65(c_void_p(buf399.data_ptr()), c_void_p(mul_779.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_598.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf398.data_ptr()))
    del buf397
    del convolution
    del mul_779
    del primals_1
    del squeeze_1
    del unsqueeze_598
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf400 = aten.convolution_backward(buf399, primals_263, primals_85, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf399
    del primals_263
    del primals_85
    buf401 = buf400[1]
    return (buf398, buf396, buf391, buf389, buf385, buf382, buf377, buf375, buf370, buf367, buf351, buf343, buf345, buf343, buf337, buf335, buf330, buf327, buf312, buf309, buf304, buf302, buf297, buf294, buf279, buf270, buf273, buf270, buf265, buf263, buf258, buf255, buf239, buf237, buf231, buf229, reinterpret_tensor(buf220, (63, 32), (32, 1), 0), reinterpret_tensor(buf216, (63, 32), (32, 1), 0), buf209, buf207, buf203, buf200, buf195, buf193, buf188, buf185, buf170, buf161, buf164, buf161, buf156, buf154, buf149, buf146, buf130, buf128, buf122, buf120, reinterpret_tensor(buf111, (31, 64), (64, 1), 0), reinterpret_tensor(buf107, (31, 64), (64, 1), 0), buf100, buf98, buf94, buf91, buf86, buf84, reinterpret_tensor(buf75, (31, 128), (128, 1), 0), reinterpret_tensor(buf71, (31, 128), (128, 1), 0), buf63, buf61, buf56, buf46, buf49, buf46, buf41, buf39, reinterpret_tensor(buf30, (15, 128), (128, 1), 0), reinterpret_tensor(buf26, (15, 128), (128, 1), 0), buf19, buf17, buf12, buf10, buf5, buf3, buf401, buf395, buf388, buf381, buf374, buf365, buf366, buf360, buf361, buf355, buf349, buf341, buf334, buf325, buf326, buf320, buf321, buf315, buf308, buf301, buf292, buf293, buf287, buf288, buf282, buf276, buf269, buf262, buf253, buf254, buf248, buf249, buf243, buf235, buf228, buf206, buf199, buf192, buf183, buf184, buf178, buf179, buf173, buf167, buf160, buf153, buf144, buf145, buf139, buf140, buf134, buf126, buf119, buf97, buf90, buf83, buf60, buf53, buf45, buf38, buf16, buf9, reinterpret_tensor(buf1, (1000, 1280), (1280, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


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
    primals_35 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((24, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((32, 24, 3, 3), (216, 1, 72, 24), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((64, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((64, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((128, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((128, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((384, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((16, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((256, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((16, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((256, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((768, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((1536, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((512, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((1280, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 24, 128, 128), (393216, 1, 3072, 24), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    mul_7 = rand_strided((8, 24, 128, 128), (393216, 1, 3072, 24), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    mul_15 = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    mul_23 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    mul_31 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_24 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    mean = rand_strided((8, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    relu = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    mul_40 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    mul_55 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    mul_63 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    add_45 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    mean_1 = rand_strided((8, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    mul_72 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    mul_80 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    mul_88 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_61 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    mean_2 = rand_strided((8, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    mul_97 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    mul_112 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    mul_120 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    add_82 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    mean_3 = rand_strided((8, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    mul_129 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    mul_137 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    mul_145 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    view_7 = rand_strided((32768, 32), (32, 1), device='cpu', dtype=torch.float32)
    view_13 = rand_strided((32768, 32), (32, 1), device='cpu', dtype=torch.float32)
    bmm_1 = rand_strided((32, 1024, 32), (32768, 32, 1), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    mul_154 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    mul_162 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    mul_170 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    add_116 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    mean_4 = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    mul_179 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    mul_194 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    mul_202 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    add_137 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    mean_5 = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cpu', dtype=torch.float32)
    mul_211 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    mul_219 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    mul_227 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    view_31 = rand_strided((8192, 64), (64, 1), device='cpu', dtype=torch.float32)
    view_37 = rand_strided((8192, 64), (64, 1), device='cpu', dtype=torch.float32)
    bmm_3 = rand_strided((32, 256, 64), (16384, 64, 1), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    mul_236 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    mul_244 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    mul_252 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    view_55 = rand_strided((8192, 128), (128, 1), device='cpu', dtype=torch.float32)
    view_61 = rand_strided((8192, 128), (128, 1), device='cpu', dtype=torch.float32)
    view_71 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    avg_pool2d = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    mul_261 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    mul_276 = rand_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    mul_284 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    view_79 = rand_strided((2048, 128), (128, 1), device='cpu', dtype=torch.float32)
    view_85 = rand_strided((2048, 128), (128, 1), device='cpu', dtype=torch.float32)
    bmm_7 = rand_strided((32, 64, 128), (8192, 128, 1), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    mul_293 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    mul_301 = rand_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((8, 1280, 8, 8), (81920, 1, 10240, 1280), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    clone_62 = rand_strided((8, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    permute_33 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    mul_311 = rand_strided((8, 1280, 8, 8), (81920, 1, 10240, 1280), device='cpu', dtype=torch.float32)
    unsqueeze_154 = rand_strided((1, 1280, 1, 1), (1280, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_323 = rand_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cpu', dtype=torch.float32)
    unsqueeze_166 = rand_strided((1, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_335 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    unsqueeze_178 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    permute_41 = rand_strided((32, 64, 64), (4096, 1, 64), device='cpu', dtype=torch.float32)
    permute_42 = rand_strided((32, 128, 64), (8192, 64, 1), device='cpu', dtype=torch.float32)
    alias_16 = rand_strided((32, 64, 64), (4096, 64, 1), device='cpu', dtype=torch.float32)
    permute_46 = rand_strided((15, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_52 = rand_strided((15, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_54 = rand_strided((32, 128, 64), (8192, 64, 1), device='cpu', dtype=torch.float32)
    permute_55 = rand_strided((32, 64, 128), (8192, 1, 64), device='cpu', dtype=torch.float32)
    mul_350 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    unsqueeze_190 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_362 = rand_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cpu', dtype=torch.float32)
    unsqueeze_202 = rand_strided((1, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_214 = rand_strided((1, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_383 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    unsqueeze_226 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    permute_62 = rand_strided((32, 256, 256), (65536, 1, 256), device='cpu', dtype=torch.float32)
    permute_63 = rand_strided((32, 128, 256), (32768, 256, 1), device='cpu', dtype=torch.float32)
    alias_17 = rand_strided((32, 256, 256), (65536, 256, 1), device='cpu', dtype=torch.float32)
    permute_67 = rand_strided((31, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_73 = rand_strided((31, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_75 = rand_strided((32, 128, 256), (32768, 256, 1), device='cpu', dtype=torch.float32)
    permute_76 = rand_strided((32, 256, 128), (32768, 1, 256), device='cpu', dtype=torch.float32)
    mul_398 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    unsqueeze_238 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_410 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_422 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    permute_83 = rand_strided((32, 256, 256), (65536, 1, 256), device='cpu', dtype=torch.float32)
    permute_84 = rand_strided((32, 64, 256), (16384, 256, 1), device='cpu', dtype=torch.float32)
    alias_18 = rand_strided((32, 256, 256), (65536, 256, 1), device='cpu', dtype=torch.float32)
    permute_88 = rand_strided((31, 64), (64, 1), device='cpu', dtype=torch.float32)
    permute_94 = rand_strided((31, 64), (64, 1), device='cpu', dtype=torch.float32)
    permute_96 = rand_strided((32, 64, 256), (16384, 256, 1), device='cpu', dtype=torch.float32)
    permute_97 = rand_strided((32, 256, 64), (16384, 1, 256), device='cpu', dtype=torch.float32)
    mul_437 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    unsqueeze_274 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_449 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    unsqueeze_286 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_477 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_489 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_526 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cpu', dtype=torch.float32)
    unsqueeze_358 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_538 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    unsqueeze_370 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_550 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    unsqueeze_382 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    permute_110 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_111 = rand_strided((32, 32, 1024), (32768, 1024, 1), device='cpu', dtype=torch.float32)
    alias_27 = rand_strided((32, 1024, 1024), (1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_115 = rand_strided((63, 32), (32, 1), device='cpu', dtype=torch.float32)
    permute_121 = rand_strided((63, 32), (32, 1), device='cpu', dtype=torch.float32)
    permute_123 = rand_strided((32, 32, 1024), (32768, 1024, 1), device='cpu', dtype=torch.float32)
    permute_124 = rand_strided((32, 1024, 32), (32768, 1, 1024), device='cpu', dtype=torch.float32)
    mul_565 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_577 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    unsqueeze_406 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_418 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_605 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cpu', dtype=torch.float32)
    unsqueeze_430 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_617 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_454 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_466 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_654 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    unsqueeze_478 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_666 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_502 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_694 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    unsqueeze_514 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_706 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    unsqueeze_526 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_538 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_550 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_743 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    unsqueeze_562 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_755 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cpu', dtype=torch.float32)
    unsqueeze_574 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_767 = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cpu', dtype=torch.float32)
    unsqueeze_586 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_779 = rand_strided((8, 24, 128, 128), (393216, 1, 3072, 24), device='cpu', dtype=torch.float32)
    unsqueeze_598 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_61, primals_63, primals_65, primals_69, primals_71, primals_73, primals_75, primals_79, primals_81, primals_83, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_92, primals_94, primals_95, primals_96, primals_97, primals_98, primals_100, primals_102, primals_103, primals_104, primals_105, primals_107, primals_109, primals_110, primals_111, primals_112, primals_113, primals_115, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_125, primals_127, primals_128, primals_129, primals_130, primals_131, primals_133, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_263, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, convolution_3, squeeze_10, mul_31, convolution_4, squeeze_13, add_24, mean, relu, convolution_6, mul_40, convolution_7, squeeze_16, convolution_8, squeeze_19, mul_55, convolution_9, squeeze_22, mul_63, convolution_10, squeeze_25, add_45, mean_1, relu_1, convolution_12, mul_72, convolution_13, squeeze_28, mul_80, convolution_14, squeeze_31, mul_88, convolution_15, squeeze_34, add_61, mean_2, relu_2, convolution_17, mul_97, convolution_18, squeeze_37, convolution_19, squeeze_40, mul_112, convolution_20, squeeze_43, mul_120, convolution_21, squeeze_46, add_82, mean_3, relu_3, convolution_23, mul_129, convolution_24, squeeze_49, mul_137, convolution_25, squeeze_52, mul_145, view_7, view_13, bmm_1, squeeze_55, mul_154, convolution_27, squeeze_58, mul_162, convolution_28, squeeze_61, mul_170, convolution_29, squeeze_64, add_116, mean_4, relu_4, convolution_31, mul_179, convolution_32, squeeze_67, convolution_33, squeeze_70, mul_194, convolution_34, squeeze_73, mul_202, convolution_35, squeeze_76, add_137, mean_5, relu_5, convolution_37, mul_211, convolution_38, squeeze_79, mul_219, convolution_39, squeeze_82, mul_227, view_31, view_37, bmm_3, squeeze_85, mul_236, convolution_41, squeeze_88, mul_244, convolution_42, squeeze_91, mul_252, view_55, view_61, view_71, avg_pool2d, squeeze_94, mul_261, convolution_44, squeeze_97, convolution_45, squeeze_100, mul_276, convolution_46, squeeze_103, mul_284, view_79, view_85, bmm_7, squeeze_106, mul_293, convolution_48, squeeze_109, mul_301, convolution_49, squeeze_112, clone_62, permute_33, mul_311, unsqueeze_154, mul_323, unsqueeze_166, mul_335, unsqueeze_178, permute_41, permute_42, alias_16, permute_46, permute_52, permute_54, permute_55, mul_350, unsqueeze_190, mul_362, unsqueeze_202, unsqueeze_214, mul_383, unsqueeze_226, permute_62, permute_63, alias_17, permute_67, permute_73, permute_75, permute_76, mul_398, unsqueeze_238, mul_410, unsqueeze_250, mul_422, unsqueeze_262, permute_83, permute_84, alias_18, permute_88, permute_94, permute_96, permute_97, mul_437, unsqueeze_274, mul_449, unsqueeze_286, unsqueeze_298, mul_477, unsqueeze_310, mul_489, unsqueeze_322, unsqueeze_334, unsqueeze_346, mul_526, unsqueeze_358, mul_538, unsqueeze_370, mul_550, unsqueeze_382, permute_110, permute_111, alias_27, permute_115, permute_121, permute_123, permute_124, mul_565, unsqueeze_394, mul_577, unsqueeze_406, unsqueeze_418, mul_605, unsqueeze_430, mul_617, unsqueeze_442, unsqueeze_454, unsqueeze_466, mul_654, unsqueeze_478, mul_666, unsqueeze_490, unsqueeze_502, mul_694, unsqueeze_514, mul_706, unsqueeze_526, unsqueeze_538, unsqueeze_550, mul_743, unsqueeze_562, mul_755, unsqueeze_574, mul_767, unsqueeze_586, mul_779, unsqueeze_598, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('sebotnet33ts_256', benchmark_compiled_module)
