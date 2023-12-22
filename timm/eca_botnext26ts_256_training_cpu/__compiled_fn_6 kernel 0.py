
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
                    auto tmp7 = static_cast<float>(0.25);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_5 = async_compile.cpp('''
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(640L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x3);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(64);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((16L*x1) + (128L*x2) + (128L*x2_inner) + (1024L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3) + (4096L*x0)), 1024L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 64L)) % static_cast<long>(16L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((16L*x2) + (16L*x2_inner) + (128L*x1) + (1024L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3) + (4096L*x0)), 1024L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 64L)) % static_cast<long>(16L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp8 = tmp6 + tmp7;
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((16L*x2) + (16L*x2_inner) + (128L*x1) + (1024L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3) + (4096L*x0)), 1024L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 64L)) % static_cast<long>(16L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp10 = tmp8 + tmp9;
                                return tmp10;
                            }
                            ;
                            auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp12 = tmp0 >= tmp3;
                            auto tmp13 = static_cast<int>(128);
                            auto tmp14 = tmp0 < tmp13;
                            auto tmp15 = tmp12 & tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr3 + static_cast<long>((-4096L) + x2 + (8L*x1) + (64L*x3) + (4096L*x0)), to_float_mask(tmp15));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp15));
                            auto tmp19 = tmp0 >= tmp13;
                            auto tmp20 = static_cast<int>(640);
                            auto tmp21 = tmp0 < tmp20;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr4[static_cast<long>((128L*x2) + (128L*x2_inner) + (1024L*x1) + (8192L*(static_cast<long>(c10::div_floor_integer(((-8192L) + x2 + x2_inner + (8L*x1) + (64L*x3) + (32768L*x0)), 8192L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (8L*x1) + (64L*x3)), 64L)) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp19)); })();
                                return tmp23;
                            }
                            ;
                            auto tmp24 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp19));
                            auto tmp25 = to_float_mask(tmp15);
                            auto tmp26 = decltype(tmp18)::blendv(tmp24, tmp18, tmp25);
                            auto tmp27 = to_float_mask(tmp4);
                            auto tmp28 = decltype(tmp11)::blendv(tmp26, tmp11, tmp27);
                            tmp28.store(out_ptr0 + static_cast<long>(x2 + (8L*x1) + (64L*x3) + (40960L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_6 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_7 = async_compile.cpp('''
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


cpp_fused_convolution_backward_native_batch_norm_backward_8 = async_compile.cpp('''
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


cpp_fused_avg_pool2d_backward_mul_native_batch_norm_backward_9 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_sum_view_10 = async_compile.cpp('''
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


cpp_fused_sum_view_11 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_mul_12 = async_compile.cpp('''
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
                    auto tmp7 = static_cast<float>(0.25);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_13 = async_compile.cpp('''
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(640L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x3);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(64);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((16L*x1) + (256L*x2) + (256L*x2_inner) + (4096L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3) + (16384L*x0)), 4096L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(16L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((16L*x2) + (16L*x2_inner) + (256L*x1) + (4096L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3) + (16384L*x0)), 4096L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(16L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp8 = tmp6 + tmp7;
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((16L*x2) + (16L*x2_inner) + (256L*x1) + (4096L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3) + (16384L*x0)), 4096L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(16L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp10 = tmp8 + tmp9;
                                return tmp10;
                            }
                            ;
                            auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp12 = tmp0 >= tmp3;
                            auto tmp13 = static_cast<int>(128);
                            auto tmp14 = tmp0 < tmp13;
                            auto tmp15 = tmp12 & tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr3 + static_cast<long>((-16384L) + x2 + (16L*x1) + (256L*x3) + (16384L*x0)), to_float_mask(tmp15));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp15));
                            auto tmp19 = tmp0 >= tmp13;
                            auto tmp20 = static_cast<int>(640);
                            auto tmp21 = tmp0 < tmp20;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr4[static_cast<long>((128L*x2) + (128L*x2_inner) + (2048L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(((-32768L) + x2 + x2_inner + (16L*x1) + (256L*x3) + (131072L*x0)), 32768L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(128L)))]; return masked_load(tmpbuf, to_float_mask(tmp19)); })();
                                return tmp23;
                            }
                            ;
                            auto tmp24 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp19));
                            auto tmp25 = to_float_mask(tmp15);
                            auto tmp26 = decltype(tmp18)::blendv(tmp24, tmp18, tmp25);
                            auto tmp27 = to_float_mask(tmp4);
                            auto tmp28 = decltype(tmp11)::blendv(tmp26, tmp11, tmp27);
                            tmp28.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (256L*x3) + (163840L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_14 = async_compile.cpp('''
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


cpp_fused_add_mul_native_batch_norm_backward_15 = async_compile.cpp('''
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


cpp_fused_mul_native_batch_norm_backward_16 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_sum_view_17 = async_compile.cpp('''
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


cpp_fused_sum_view_18 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_mul_19 = async_compile.cpp('''
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
                    auto tmp7 = static_cast<float>(0.25);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_backward_20 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x3);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(64);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((16L*x1) + (256L*x2) + (256L*x2_inner) + (4096L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3) + (16384L*x0)), 4096L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(16L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((16L*x2) + (16L*x2_inner) + (256L*x1) + (4096L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3) + (16384L*x0)), 4096L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(16L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp8 = tmp6 + tmp7;
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((16L*x2) + (16L*x2_inner) + (256L*x1) + (4096L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3) + (16384L*x0)), 4096L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(16L)))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                                auto tmp10 = tmp8 + tmp9;
                                return tmp10;
                            }
                            ;
                            auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp12 = tmp0 >= tmp3;
                            auto tmp13 = static_cast<int>(128);
                            auto tmp14 = tmp0 < tmp13;
                            auto tmp15 = tmp12 & tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr3 + static_cast<long>((-16384L) + x2 + (16L*x1) + (256L*x3) + (16384L*x0)), to_float_mask(tmp15));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp15));
                            auto tmp19 = tmp0 >= tmp13;
                            auto tmp20 = static_cast<int>(384);
                            auto tmp21 = tmp0 < tmp20;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr4[static_cast<long>((64L*x2) + (64L*x2_inner) + (1024L*x1) + (16384L*(static_cast<long>(c10::div_floor_integer(((-32768L) + x2 + x2_inner + (16L*x1) + (256L*x3) + (65536L*x0)), 16384L)) % static_cast<long>(32L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner + (16L*x1) + (256L*x3)), 256L)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp19)); })();
                                return tmp23;
                            }
                            ;
                            auto tmp24 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp19));
                            auto tmp25 = to_float_mask(tmp15);
                            auto tmp26 = decltype(tmp18)::blendv(tmp24, tmp18, tmp25);
                            auto tmp27 = to_float_mask(tmp4);
                            auto tmp28 = decltype(tmp11)::blendv(tmp26, tmp11, tmp27);
                            tmp28.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (256L*x3) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_21 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_22 = async_compile.cpp('''
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


cpp_fused_native_batch_norm_backward_23 = async_compile.cpp('''
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


cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_25 = async_compile.cpp('''
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


cpp_fused_convolution_backward_mul_native_batch_norm_backward_26 = async_compile.cpp('''
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


cpp_fused_add_mul_native_batch_norm_backward_27 = async_compile.cpp('''
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


cpp_fused_native_batch_norm_backward_32 = async_compile.cpp('''
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


cpp_fused_native_batch_norm_backward_41 = async_compile.cpp('''
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


cpp_fused_add_45 = async_compile.cpp('''
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


cpp_fused_convolution_backward_mul_native_batch_norm_backward_46 = async_compile.cpp('''
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


cpp_fused_convolution_backward_mul_native_batch_norm_backward_47 = async_compile.cpp('''
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
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_47, primals_49, primals_51, primals_55, primals_57, primals_59, primals_61, primals_65, primals_67, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_200, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, getitem_6, getitem_7, convolution_3, squeeze_10, mul_31, convolution_4, squeeze_13, add_24, view, convolution_5, mul_40, convolution_6, squeeze_16, convolution_7, squeeze_19, mul_55, convolution_8, squeeze_22, mul_63, convolution_9, squeeze_25, add_45, view_2, convolution_10, mul_72, convolution_11, squeeze_28, mul_80, convolution_12, squeeze_31, mul_88, convolution_13, squeeze_34, add_61, view_4, convolution_14, mul_97, convolution_15, squeeze_37, convolution_16, squeeze_40, mul_112, convolution_17, squeeze_43, mul_120, convolution_18, squeeze_46, add_82, view_6, convolution_19, mul_129, convolution_20, squeeze_49, mul_137, convolution_21, squeeze_52, mul_145, convolution_22, squeeze_55, add_98, view_8, convolution_23, mul_154, convolution_24, squeeze_58, convolution_25, squeeze_61, mul_169, convolution_26, squeeze_64, mul_177, view_17, view_23, bmm_1, squeeze_67, mul_186, convolution_28, squeeze_70, mul_194, convolution_29, squeeze_73, mul_202, view_41, view_47, view_57, avg_pool2d, squeeze_76, mul_211, convolution_31, squeeze_79, convolution_32, squeeze_82, mul_226, convolution_33, squeeze_85, mul_234, view_65, view_71, bmm_5, squeeze_88, mul_243, convolution_35, squeeze_91, clone_48, permute_25, mul_253, unsqueeze_126, mul_265, unsqueeze_138, permute_32, permute_33, alias_8, permute_37, permute_43, permute_45, permute_46, mul_280, unsqueeze_150, mul_292, unsqueeze_162, unsqueeze_174, mul_313, unsqueeze_186, permute_53, permute_54, alias_9, permute_58, permute_64, permute_66, permute_67, mul_328, unsqueeze_198, mul_340, unsqueeze_210, mul_352, unsqueeze_222, permute_74, permute_75, alias_10, permute_79, permute_85, permute_87, permute_88, mul_367, unsqueeze_234, mul_379, unsqueeze_246, unsqueeze_258, unsqueeze_272, mul_416, unsqueeze_284, mul_428, unsqueeze_296, unsqueeze_310, mul_456, unsqueeze_322, mul_468, unsqueeze_334, unsqueeze_346, unsqueeze_360, mul_505, unsqueeze_372, mul_517, unsqueeze_384, unsqueeze_398, mul_545, unsqueeze_410, mul_557, unsqueeze_422, unsqueeze_434, unsqueeze_448, mul_594, unsqueeze_460, mul_606, unsqueeze_472, mul_618, unsqueeze_484, mul_630, unsqueeze_496, tangents_1 = args
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
    assert_size_stride(primals_96, (384, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_97, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_98, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_99, (640, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_100, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_101, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_102, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_103, (640, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_104, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_200, (8, 3, 256, 256), (196608, 1, 768, 3))
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
    assert_size_stride(view_17, (8192, 16), (16, 1))
    assert_size_stride(view_23, (8192, 16), (16, 1))
    assert_size_stride(bmm_1, (32, 256, 64), (16384, 64, 1))
    assert_size_stride(squeeze_67, (256, ), (1, ))
    assert_size_stride(mul_186, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_28, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(squeeze_70, (1024, ), (1, ))
    assert_size_stride(mul_194, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(convolution_29, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(squeeze_73, (512, ), (1, ))
    assert_size_stride(mul_202, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(view_41, (8192, 16), (16, 1))
    assert_size_stride(view_47, (8192, 16), (16, 1))
    assert_size_stride(view_57, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(avg_pool2d, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_76, (512, ), (1, ))
    assert_size_stride(mul_211, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_31, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(squeeze_79, (2048, ), (1, ))
    assert_size_stride(convolution_32, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(squeeze_82, (2048, ), (1, ))
    assert_size_stride(mul_226, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(convolution_33, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_85, (512, ), (1, ))
    assert_size_stride(mul_234, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(view_65, (2048, 16), (16, 1))
    assert_size_stride(view_71, (2048, 16), (16, 1))
    assert_size_stride(bmm_5, (32, 64, 128), (8192, 128, 1))
    assert_size_stride(squeeze_88, (512, ), (1, ))
    assert_size_stride(mul_243, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_35, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(squeeze_91, (2048, ), (1, ))
    assert_size_stride(clone_48, (8, 2048), (2048, 1))
    assert_size_stride(permute_25, (1000, 2048), (2048, 1))
    assert_size_stride(mul_253, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(unsqueeze_126, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(mul_265, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_138, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(permute_32, (32, 64, 64), (4096, 1, 64))
    assert_size_stride(permute_33, (32, 128, 64), (8192, 64, 1))
    assert_size_stride(alias_8, (32, 64, 64), (4096, 64, 1))
    assert_size_stride(permute_37, (15, 16), (16, 1))
    assert_size_stride(permute_43, (15, 16), (16, 1))
    assert_size_stride(permute_45, (32, 16, 64), (1024, 64, 1))
    assert_size_stride(permute_46, (32, 64, 16), (1024, 1, 64))
    assert_size_stride(mul_280, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_150, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_292, (8, 2048, 8, 8), (131072, 1, 16384, 2048))
    assert_size_stride(unsqueeze_162, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_174, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(mul_313, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_186, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(permute_53, (32, 256, 256), (65536, 1, 256))
    assert_size_stride(permute_54, (32, 128, 256), (32768, 256, 1))
    assert_size_stride(alias_9, (32, 256, 256), (65536, 256, 1))
    assert_size_stride(permute_58, (31, 16), (16, 1))
    assert_size_stride(permute_64, (31, 16), (16, 1))
    assert_size_stride(permute_66, (32, 16, 256), (4096, 256, 1))
    assert_size_stride(permute_67, (32, 256, 16), (4096, 1, 256))
    assert_size_stride(mul_328, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(unsqueeze_198, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_340, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(unsqueeze_210, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(mul_352, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_222, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(permute_74, (32, 256, 256), (65536, 1, 256))
    assert_size_stride(permute_75, (32, 64, 256), (16384, 256, 1))
    assert_size_stride(alias_10, (32, 256, 256), (65536, 256, 1))
    assert_size_stride(permute_79, (31, 16), (16, 1))
    assert_size_stride(permute_85, (31, 16), (16, 1))
    assert_size_stride(permute_87, (32, 16, 256), (4096, 256, 1))
    assert_size_stride(permute_88, (32, 256, 16), (4096, 1, 256))
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
    extern_kernels.mm(tangents_1, permute_25, out=buf0)
    del permute_25
    buf1 = empty((1000, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_48, out=buf1)
    del clone_48
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf4 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf5 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf6 = empty((8, 2048, 8, 8), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_mul_native_batch_norm_backward_sum_0(c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(mul_253.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_126.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del convolution_35
    del primals_67
    del squeeze_91
    del tangents_1
    del unsqueeze_126
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
    buf7 = aten.convolution_backward(buf6, mul_243, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_243
    del primals_104
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf10 = empty((512, ), device='cpu', dtype=torch.float32)
    buf11 = empty((512, ), device='cpu', dtype=torch.float32)
    buf12 = empty((512, ), device='cpu', dtype=torch.float32)
    buf13 = empty((8, 512, 8, 8), device='cpu', dtype=torch.float32)
    cpp_fused_mul_native_batch_norm_backward_1(c_void_p(buf8.data_ptr()), c_void_p(mul_265.data_ptr()), c_void_p(bmm_5.data_ptr()), c_void_p(unsqueeze_138.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    del bmm_5
    del mul_265
    del primals_65
    del squeeze_88
    del unsqueeze_138
    buf14 = reinterpret_tensor(buf8, (32, 64, 128), (8192, 128, 1), 0); del buf8  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_32, reinterpret_tensor(buf13, (32, 64, 128), (8192, 1, 64), 0), out=buf14)
    del permute_32
    buf15 = empty((32, 64, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf13, (32, 64, 128), (8192, 1, 64), 0), permute_33, out=buf15)
    del buf13
    del permute_33
    buf16 = reinterpret_tensor(buf4, (32, 64, 1), (64, 1, 2048), 0); del buf4  # reuse
    buf17 = empty((32, 8, 1, 8, 8), device='cpu', dtype=torch.float32)
    buf18 = empty((2048, 15), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_sum_view_2(c_void_p(buf15.data_ptr()), c_void_p(alias_8.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()))
    buf19 = empty((15, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf18, (15, 2048), (1, 15), 0), view_71, out=buf19)
    del view_71
    buf20 = empty((2048, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf18, permute_37, out=buf20)
    del permute_37
    buf21 = buf17; del buf17  # reuse
    buf22 = buf18; del buf18  # reuse
    cpp_fused_sum_view_3(c_void_p(buf15.data_ptr()), c_void_p(alias_8.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()))
    del buf21
    buf23 = empty((15, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf22, (15, 2048), (1, 15), 0), view_65, out=buf23)
    del view_65
    buf24 = empty((2048, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf22, permute_43, out=buf24)
    del buf22
    del permute_43
    buf25 = buf15; del buf15  # reuse
    cpp_fused__softmax_backward_data_mul_4(c_void_p(buf25.data_ptr()), c_void_p(alias_8.data_ptr()), c_void_p(buf16.data_ptr()))
    del alias_8
    buf26 = empty((32, 16, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_45, buf25, out=buf26)
    del permute_45
    buf27 = empty((32, 64, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf25, permute_46, out=buf27)
    del permute_46
    buf28 = empty((8, 640, 8, 8), device='cpu', dtype=torch.float32)
    cpp_fused_cat_convolution_backward_5(c_void_p(buf20.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf28.data_ptr()))
    del buf14
    del buf20
    del buf24
    del buf26
    del buf27
    # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]
    buf29 = aten.convolution_backward(buf28, mul_234, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf28
    del mul_234
    del primals_103
    buf30 = buf29[0]
    buf31 = buf29[1]
    del buf29
    buf32 = buf11; del buf11  # reuse
    buf33 = empty((512, ), device='cpu', dtype=torch.float32)
    buf34 = empty((512, ), device='cpu', dtype=torch.float32)
    buf35 = buf30; del buf30  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_6(c_void_p(buf35.data_ptr()), c_void_p(mul_280.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_150.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()))
    del convolution_33
    del mul_280
    del primals_61
    del squeeze_85
    del unsqueeze_150
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf36 = aten.convolution_backward(buf35, mul_226, primals_102, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf35
    del mul_226
    del primals_102
    buf37 = buf36[0]
    buf38 = buf36[1]
    del buf36
    buf39 = reinterpret_tensor(buf16, (2048, ), (1, ), 0); del buf16  # reuse
    buf40 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf47 = empty((2048, ), device='cpu', dtype=torch.float32)
    buf41 = reinterpret_tensor(buf6, (8, 2048, 8, 8), (131072, 1, 16384, 2048), 0); del buf6  # reuse
    buf48 = empty_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    buf42 = buf40; del buf40  # reuse
    buf43 = empty((8, 2048, 8, 8), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_7(c_void_p(buf42.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(mul_253.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(mul_292.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(unsqueeze_162.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(unsqueeze_174.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf43.data_ptr()))
    del buf0
    del buf37
    del buf41
    del convolution_31
    del convolution_32
    del mul_253
    del mul_292
    del primals_59
    del squeeze_82
    del unsqueeze_162
    del unsqueeze_174
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf44 = aten.convolution_backward(buf43, mul_194, primals_101, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_101
    buf45 = buf44[0]
    buf46 = buf44[1]
    del buf44
    buf49 = buf47; del buf47  # reuse
    buf50 = buf43; del buf43  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_8(c_void_p(buf49.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf50.data_ptr()))
    del primals_57
    del squeeze_79
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf51 = aten.convolution_backward(buf50, mul_211, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_211
    del primals_100
    buf52 = buf51[0]
    buf53 = buf51[1]
    del buf51
    buf54 = buf33; del buf33  # reuse
    buf55 = empty((512, ), device='cpu', dtype=torch.float32)
    buf56 = empty((512, ), device='cpu', dtype=torch.float32)
    buf57 = buf52; del buf52  # reuse
    buf58 = reinterpret_tensor(buf50, (8, 512, 16, 16), (131072, 256, 16, 1), 0); del buf50  # reuse
    cpp_fused_avg_pool2d_backward_mul_native_batch_norm_backward_9(c_void_p(buf57.data_ptr()), c_void_p(mul_313.data_ptr()), c_void_p(avg_pool2d.data_ptr()), c_void_p(unsqueeze_186.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf58.data_ptr()))
    del avg_pool2d
    del buf57
    del mul_313
    del primals_55
    del squeeze_76
    del unsqueeze_186
    buf59 = reinterpret_tensor(buf48, (32, 256, 128), (32768, 128, 1), 0); del buf48  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_53, reinterpret_tensor(buf58, (32, 256, 128), (32768, 1, 256), 0), out=buf59)
    del permute_53
    buf60 = empty((32, 256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf58, (32, 256, 128), (32768, 1, 256), 0), permute_54, out=buf60)
    del buf58
    del permute_54
    buf61 = empty_strided((32, 256, 1), (256, 1, 8192), device='cpu', dtype=torch.float32)
    buf62 = reinterpret_tensor(buf25, (32, 16, 1, 16, 16), (4096, 256, 256, 16, 1), 0); del buf25  # reuse
    buf63 = empty((8192, 31), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_sum_view_10(c_void_p(buf60.data_ptr()), c_void_p(alias_9.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()))
    buf64 = empty((31, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf63, (31, 8192), (1, 31), 0), view_47, out=buf64)
    del view_47
    buf65 = reinterpret_tensor(buf62, (8192, 16), (16, 1), 0); del buf62  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf63, permute_58, out=buf65)
    del permute_58
    buf66 = empty((32, 16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf67 = buf63; del buf63  # reuse
    cpp_fused_sum_view_11(c_void_p(buf60.data_ptr()), c_void_p(alias_9.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()))
    buf68 = empty((31, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf67, (31, 8192), (1, 31), 0), view_41, out=buf68)
    del view_41
    buf69 = reinterpret_tensor(buf66, (8192, 16), (16, 1), 0); del buf66  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf67, permute_64, out=buf69)
    del permute_64
    buf70 = buf60; del buf60  # reuse
    cpp_fused__softmax_backward_data_mul_12(c_void_p(buf70.data_ptr()), c_void_p(alias_9.data_ptr()), c_void_p(buf61.data_ptr()))
    del alias_9
    buf71 = empty((32, 16, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_66, buf70, out=buf71)
    del permute_66
    buf72 = empty((32, 256, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf70, permute_67, out=buf72)
    del permute_67
    buf73 = empty((8, 640, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused_cat_convolution_backward_13(c_void_p(buf65.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf73.data_ptr()))
    del buf59
    # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]
    buf74 = aten.convolution_backward(buf73, mul_202, primals_99, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf73
    del mul_202
    del primals_99
    buf75 = buf74[0]
    buf76 = buf74[1]
    del buf74
    buf77 = buf55; del buf55  # reuse
    buf78 = empty((512, ), device='cpu', dtype=torch.float32)
    buf79 = empty((512, ), device='cpu', dtype=torch.float32)
    buf80 = buf75; del buf75  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_14(c_void_p(buf80.data_ptr()), c_void_p(mul_328.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_198.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    del convolution_29
    del mul_328
    del primals_51
    del squeeze_73
    del unsqueeze_198
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf81 = aten.convolution_backward(buf80, mul_194, primals_98, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf80
    del mul_194
    del primals_98
    buf82 = buf81[0]
    buf83 = buf81[1]
    del buf81
    buf84 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf85 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf86 = reinterpret_tensor(buf70, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf70  # reuse
    buf87 = buf85; del buf85  # reuse
    cpp_fused_add_mul_native_batch_norm_backward_15(c_void_p(buf87.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(mul_340.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(unsqueeze_210.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf86.data_ptr()))
    del convolution_28
    del primals_49
    del squeeze_70
    del unsqueeze_210
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf88 = aten.convolution_backward(buf86, mul_186, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_186
    del primals_97
    buf89 = buf88[0]
    buf90 = buf88[1]
    del buf88
    buf91 = empty((256, ), device='cpu', dtype=torch.float32)
    buf92 = empty((256, ), device='cpu', dtype=torch.float32)
    buf93 = empty((256, ), device='cpu', dtype=torch.float32)
    buf94 = empty((8, 256, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused_mul_native_batch_norm_backward_16(c_void_p(buf89.data_ptr()), c_void_p(mul_352.data_ptr()), c_void_p(bmm_1.data_ptr()), c_void_p(unsqueeze_222.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()))
    del bmm_1
    del mul_352
    del primals_47
    del squeeze_67
    del unsqueeze_222
    buf95 = reinterpret_tensor(buf89, (32, 256, 64), (16384, 64, 1), 0); del buf89  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_74, reinterpret_tensor(buf94, (32, 256, 64), (16384, 1, 256), 0), out=buf95)
    del permute_74
    buf96 = reinterpret_tensor(buf86, (32, 256, 256), (65536, 256, 1), 0); del buf86  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf94, (32, 256, 64), (16384, 1, 256), 0), permute_75, out=buf96)
    del buf94
    del permute_75
    buf97 = buf61; del buf61  # reuse
    buf98 = reinterpret_tensor(buf72, (32, 16, 1, 16, 16), (4096, 256, 256, 16, 1), 0); del buf72  # reuse
    buf99 = buf67; del buf67  # reuse
    cpp_fused__softmax_backward_data_sum_view_17(c_void_p(buf96.data_ptr()), c_void_p(alias_10.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()))
    buf100 = empty((31, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf99, (31, 8192), (1, 31), 0), view_23, out=buf100)
    del view_23
    buf101 = reinterpret_tensor(buf98, (8192, 16), (16, 1), 0); del buf98  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf99, permute_79, out=buf101)
    del permute_79
    buf102 = reinterpret_tensor(buf71, (32, 16, 1, 16, 16), (4096, 256, 256, 16, 1), 0); del buf71  # reuse
    buf103 = buf99; del buf99  # reuse
    cpp_fused_sum_view_18(c_void_p(buf96.data_ptr()), c_void_p(alias_10.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()))
    buf104 = empty((31, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf103, (31, 8192), (1, 31), 0), view_17, out=buf104)
    del view_17
    buf105 = reinterpret_tensor(buf102, (8192, 16), (16, 1), 0); del buf102  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf103, permute_85, out=buf105)
    del buf103
    del permute_85
    buf106 = buf96; del buf96  # reuse
    cpp_fused__softmax_backward_data_mul_19(c_void_p(buf106.data_ptr()), c_void_p(alias_10.data_ptr()), c_void_p(buf97.data_ptr()))
    del alias_10
    del buf97
    buf107 = reinterpret_tensor(buf69, (32, 16, 256), (4096, 256, 1), 0); del buf69  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_87, buf106, out=buf107)
    del permute_87
    buf108 = reinterpret_tensor(buf65, (32, 256, 16), (4096, 16, 1), 0); del buf65  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf106, permute_88, out=buf108)
    del permute_88
    buf109 = empty((8, 384, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused_cat_convolution_backward_20(c_void_p(buf101.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf109.data_ptr()))
    del buf101
    del buf105
    del buf107
    del buf108
    del buf95
    # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]
    buf110 = aten.convolution_backward(buf109, mul_177, primals_96, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf109
    del mul_177
    del primals_96
    buf111 = buf110[0]
    buf112 = buf110[1]
    del buf110
    buf113 = buf92; del buf92  # reuse
    buf114 = empty((256, ), device='cpu', dtype=torch.float32)
    buf115 = empty((256, ), device='cpu', dtype=torch.float32)
    buf116 = buf111; del buf111  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_21(c_void_p(buf116.data_ptr()), c_void_p(mul_367.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_234.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()))
    del convolution_26
    del mul_367
    del primals_43
    del squeeze_64
    del unsqueeze_234
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf117 = aten.convolution_backward(buf116, mul_169, primals_95, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf116
    del mul_169
    del primals_95
    buf118 = buf117[0]
    buf119 = buf117[1]
    del buf117
    buf120 = buf118; del buf118  # reuse
    buf121 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf122 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf128 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf123 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf124 = reinterpret_tensor(buf106, (8, 1024, 16, 16), (262144, 1, 16384, 1024), 0); del buf106  # reuse
    buf130 = empty_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_22(c_void_p(buf120.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(mul_340.data_ptr()), c_void_p(mul_379.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_246.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_258.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf130.data_ptr()))
    del buf120
    del buf45
    del buf82
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
    buf125 = aten.convolution_backward(buf124, mul_137, primals_94, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf124
    del primals_94
    buf126 = buf125[0]
    buf127 = buf125[1]
    del buf125
    buf129 = buf128; del buf128  # reuse
    cpp_fused_native_batch_norm_backward_23(c_void_p(buf129.data_ptr()), c_void_p(squeeze_58.data_ptr()))
    del squeeze_58
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf131 = aten.convolution_backward(buf130, mul_154, primals_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf130
    del mul_154
    del primals_93
    buf132 = buf131[0]
    buf133 = buf131[1]
    del buf131
    buf134 = empty((8, 256, 1, 1), device='cpu', dtype=torch.float32)
    buf135 = reinterpret_tensor(buf134, (8, 1, 256), (256, 256, 1), 0); del buf134  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_24(c_void_p(buf135.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(add_98.data_ptr()), c_void_p(convolution_23.data_ptr()))
    # Source Nodes: [sigmoid_4], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf136 = aten.convolution_backward(buf135, view_8, primals_92, [0], [1], [2], [1], False, [0], 1, [True, True, False])
    del buf135
    del primals_92
    del view_8
    buf137 = buf136[0]
    buf138 = buf136[1]
    del buf136
    buf139 = buf114; del buf114  # reuse
    buf140 = empty((256, ), device='cpu', dtype=torch.float32)
    buf141 = buf132; del buf132  # reuse
    buf142 = buf140; del buf140  # reuse
    buf143 = buf141; del buf141  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_25(c_void_p(buf143.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(add_98.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(unsqueeze_272.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf139.data_ptr()))
    del add_98
    del buf137
    del convolution_22
    del convolution_23
    del primals_37
    del squeeze_55
    del unsqueeze_272
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf144 = aten.convolution_backward(buf143, mul_145, primals_91, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False])
    del buf143
    del mul_145
    del primals_91
    buf145 = buf144[0]
    buf146 = buf144[1]
    del buf144
    buf147 = empty((256, ), device='cpu', dtype=torch.float32)
    buf148 = empty((256, ), device='cpu', dtype=torch.float32)
    buf149 = empty((256, ), device='cpu', dtype=torch.float32)
    buf150 = buf145; del buf145  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_26(c_void_p(buf150.data_ptr()), c_void_p(mul_416.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_284.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()))
    del convolution_21
    del mul_416
    del primals_35
    del squeeze_52
    del unsqueeze_284
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf151 = aten.convolution_backward(buf150, mul_137, primals_90, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf150
    del mul_137
    del primals_90
    buf152 = buf151[0]
    buf153 = buf151[1]
    del buf151
    buf154 = buf78; del buf78  # reuse
    buf155 = empty((512, ), device='cpu', dtype=torch.float32)
    buf156 = empty_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    buf157 = buf155; del buf155  # reuse
    cpp_fused_add_mul_native_batch_norm_backward_27(c_void_p(buf157.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(mul_428.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_296.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf156.data_ptr()))
    del convolution_20
    del primals_33
    del squeeze_49
    del unsqueeze_296
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf158 = aten.convolution_backward(buf156, mul_129, primals_89, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_129
    del primals_89
    buf159 = buf158[0]
    buf160 = buf158[1]
    del buf158
    buf161 = reinterpret_tensor(buf122, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf122  # reuse
    buf162 = reinterpret_tensor(buf161, (8, 1, 128), (128, 128, 1), 0); del buf161  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_28(c_void_p(buf162.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(add_82.data_ptr()), c_void_p(convolution_19.data_ptr()))
    # Source Nodes: [sigmoid_3], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf163 = aten.convolution_backward(buf162, view_6, primals_88, [0], [1], [2], [1], False, [0], 1, [True, True, False])
    del buf162
    del primals_88
    del view_6
    buf164 = buf163[0]
    buf165 = buf163[1]
    del buf163
    buf166 = empty((128, ), device='cpu', dtype=torch.float32)
    buf167 = empty((128, ), device='cpu', dtype=torch.float32)
    buf168 = buf159; del buf159  # reuse
    buf169 = buf167; del buf167  # reuse
    buf170 = buf168; del buf168  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_29(c_void_p(buf170.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(add_82.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_310.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf166.data_ptr()))
    del add_82
    del convolution_18
    del convolution_19
    del primals_31
    del squeeze_46
    del unsqueeze_310
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf171 = aten.convolution_backward(buf170, mul_120, primals_87, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf170
    del mul_120
    del primals_87
    buf172 = buf171[0]
    buf173 = buf171[1]
    del buf171
    buf174 = empty((128, ), device='cpu', dtype=torch.float32)
    buf175 = empty((128, ), device='cpu', dtype=torch.float32)
    buf176 = empty((128, ), device='cpu', dtype=torch.float32)
    buf177 = buf172; del buf172  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_30(c_void_p(buf177.data_ptr()), c_void_p(mul_456.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(unsqueeze_322.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()))
    del convolution_17
    del mul_456
    del primals_29
    del squeeze_43
    del unsqueeze_322
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf178 = aten.convolution_backward(buf177, mul_112, primals_86, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf177
    del mul_112
    del primals_86
    buf179 = buf178[0]
    buf180 = buf178[1]
    del buf178
    buf181 = buf126; del buf126  # reuse
    buf182 = empty((512, ), device='cpu', dtype=torch.float32)
    buf183 = empty((512, ), device='cpu', dtype=torch.float32)
    buf189 = empty((512, ), device='cpu', dtype=torch.float32)
    buf184 = empty((512, ), device='cpu', dtype=torch.float32)
    buf185 = buf156; del buf156  # reuse
    buf191 = empty_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_31(c_void_p(buf181.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(mul_428.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(mul_468.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_334.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_346.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf191.data_ptr()))
    del buf152
    del buf179
    del buf181
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
    buf186 = aten.convolution_backward(buf185, mul_80, primals_85, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf185
    del primals_85
    buf187 = buf186[0]
    buf188 = buf186[1]
    del buf186
    buf190 = buf189; del buf189  # reuse
    cpp_fused_native_batch_norm_backward_32(c_void_p(buf190.data_ptr()), c_void_p(squeeze_37.data_ptr()))
    del squeeze_37
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf192 = aten.convolution_backward(buf191, mul_97, primals_84, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf191
    del mul_97
    del primals_84
    buf193 = buf192[0]
    buf194 = buf192[1]
    del buf192
    buf195 = reinterpret_tensor(buf164, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf164  # reuse
    buf196 = reinterpret_tensor(buf195, (8, 1, 128), (128, 128, 1), 0); del buf195  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_33(c_void_p(buf196.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(add_61.data_ptr()), c_void_p(convolution_14.data_ptr()))
    # Source Nodes: [sigmoid_2], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf197 = aten.convolution_backward(buf196, view_4, primals_83, [0], [1], [2], [1], False, [0], 1, [True, True, False])
    del buf196
    del primals_83
    del view_4
    buf198 = buf197[0]
    buf199 = buf197[1]
    del buf197
    buf200 = buf175; del buf175  # reuse
    buf201 = empty((128, ), device='cpu', dtype=torch.float32)
    buf202 = buf193; del buf193  # reuse
    buf203 = buf201; del buf201  # reuse
    buf204 = buf202; del buf202  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_34(c_void_p(buf204.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(add_61.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_360.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf200.data_ptr()))
    del add_61
    del buf198
    del convolution_13
    del convolution_14
    del primals_23
    del squeeze_34
    del unsqueeze_360
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf205 = aten.convolution_backward(buf204, mul_88, primals_82, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf204
    del mul_88
    del primals_82
    buf206 = buf205[0]
    buf207 = buf205[1]
    del buf205
    buf208 = empty((128, ), device='cpu', dtype=torch.float32)
    buf209 = empty((128, ), device='cpu', dtype=torch.float32)
    buf210 = empty((128, ), device='cpu', dtype=torch.float32)
    buf211 = buf206; del buf206  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_35(c_void_p(buf211.data_ptr()), c_void_p(mul_505.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_372.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()))
    del buf209
    del convolution_12
    del mul_505
    del primals_21
    del squeeze_31
    del unsqueeze_372
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf212 = aten.convolution_backward(buf211, mul_80, primals_81, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf211
    del mul_80
    del primals_81
    buf213 = buf212[0]
    buf214 = buf212[1]
    del buf212
    buf215 = buf148; del buf148  # reuse
    buf216 = empty((256, ), device='cpu', dtype=torch.float32)
    buf217 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    buf218 = buf216; del buf216  # reuse
    cpp_fused_add_mul_native_batch_norm_backward_36(c_void_p(buf218.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(mul_517.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_384.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf217.data_ptr()))
    del convolution_11
    del primals_19
    del squeeze_28
    del unsqueeze_384
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf219 = aten.convolution_backward(buf217, mul_72, primals_80, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_72
    del primals_80
    buf220 = buf219[0]
    buf221 = buf219[1]
    del buf219
    buf222 = reinterpret_tensor(buf183, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf183  # reuse
    buf223 = reinterpret_tensor(buf222, (8, 1, 64), (64, 64, 1), 0); del buf222  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_37(c_void_p(buf223.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(add_45.data_ptr()), c_void_p(convolution_10.data_ptr()))
    # Source Nodes: [sigmoid_1], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf224 = aten.convolution_backward(buf223, view_2, primals_79, [0], [1], [1], [1], False, [0], 1, [True, True, False])
    del buf223
    del primals_79
    del view_2
    buf225 = buf224[0]
    buf226 = buf224[1]
    del buf224
    buf227 = empty((64, ), device='cpu', dtype=torch.float32)
    buf228 = empty((64, ), device='cpu', dtype=torch.float32)
    buf229 = buf220; del buf220  # reuse
    buf230 = buf228; del buf228  # reuse
    buf231 = buf229; del buf229  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_38(c_void_p(buf231.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(add_45.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_398.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf227.data_ptr()))
    del add_45
    del convolution_10
    del convolution_9
    del primals_17
    del squeeze_25
    del unsqueeze_398
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf232 = aten.convolution_backward(buf231, mul_63, primals_78, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False])
    del buf231
    del mul_63
    del primals_78
    buf233 = buf232[0]
    buf234 = buf232[1]
    del buf232
    buf235 = empty((64, ), device='cpu', dtype=torch.float32)
    buf236 = empty((64, ), device='cpu', dtype=torch.float32)
    buf237 = empty((64, ), device='cpu', dtype=torch.float32)
    buf238 = buf233; del buf233  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_39(c_void_p(buf238.data_ptr()), c_void_p(mul_545.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_410.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()))
    del convolution_8
    del mul_545
    del primals_15
    del squeeze_22
    del unsqueeze_410
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf239 = aten.convolution_backward(buf238, mul_55, primals_77, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf238
    del mul_55
    del primals_77
    buf240 = buf239[0]
    buf241 = buf239[1]
    del buf239
    buf242 = buf187; del buf187  # reuse
    buf243 = empty((256, ), device='cpu', dtype=torch.float32)
    buf244 = empty((256, ), device='cpu', dtype=torch.float32)
    buf250 = empty((256, ), device='cpu', dtype=torch.float32)
    buf245 = empty((256, ), device='cpu', dtype=torch.float32)
    buf246 = buf217; del buf217  # reuse
    buf252 = empty_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_mul_native_batch_norm_backward_40(c_void_p(buf242.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(mul_517.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(mul_557.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_422.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_434.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf252.data_ptr()))
    del buf213
    del buf240
    del buf242
    del buf244
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
    buf247 = aten.convolution_backward(buf246, getitem_6, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf246
    del primals_76
    buf248 = buf247[0]
    buf249 = buf247[1]
    del buf247
    buf251 = buf250; del buf250  # reuse
    cpp_fused_native_batch_norm_backward_41(c_void_p(buf251.data_ptr()), c_void_p(squeeze_16.data_ptr()))
    del squeeze_16
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf253 = aten.convolution_backward(buf252, mul_40, primals_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf252
    del mul_40
    del primals_75
    buf254 = buf253[0]
    buf255 = buf253[1]
    del buf253
    buf256 = reinterpret_tensor(buf225, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf225  # reuse
    buf257 = reinterpret_tensor(buf256, (8, 1, 64), (64, 64, 1), 0); del buf256  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_silu_sum_42(c_void_p(buf257.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(add_24.data_ptr()), c_void_p(convolution_5.data_ptr()))
    # Source Nodes: [sigmoid], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf258 = aten.convolution_backward(buf257, view, primals_74, [0], [1], [1], [1], False, [0], 1, [True, True, False])
    del buf257
    del primals_74
    del view
    buf259 = buf258[0]
    buf260 = buf258[1]
    del buf258
    buf261 = buf236; del buf236  # reuse
    buf262 = empty((64, ), device='cpu', dtype=torch.float32)
    buf263 = buf254; del buf254  # reuse
    buf264 = buf262; del buf262  # reuse
    buf265 = buf263; del buf263  # reuse
    cpp_fused_add_convolution_backward_div_fill_mul_native_batch_norm_backward_sigmoid_sub_43(c_void_p(buf265.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(add_24.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_448.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf261.data_ptr()))
    del add_24
    del buf259
    del convolution_4
    del convolution_5
    del primals_9
    del squeeze_13
    del unsqueeze_448
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf266 = aten.convolution_backward(buf265, mul_31, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False])
    del buf265
    del mul_31
    del primals_73
    buf267 = buf266[0]
    buf268 = buf266[1]
    del buf266
    buf269 = empty((64, ), device='cpu', dtype=torch.float32)
    buf270 = empty((64, ), device='cpu', dtype=torch.float32)
    buf271 = empty((64, ), device='cpu', dtype=torch.float32)
    buf272 = buf267; del buf267  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_44(c_void_p(buf272.data_ptr()), c_void_p(mul_594.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_460.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()))
    del convolution_3
    del mul_594
    del primals_7
    del squeeze_10
    del unsqueeze_460
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf273 = aten.convolution_backward(buf272, getitem_6, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf272
    del getitem_6
    del primals_72
    buf274 = buf273[0]
    buf275 = buf273[1]
    del buf273
    buf276 = buf248; del buf248  # reuse
    cpp_fused_add_45(c_void_p(buf276.data_ptr()), c_void_p(buf274.data_ptr()))
    del buf274
    # Source Nodes: [], Original ATen: [aten.add, aten.max_pool2d_with_indices_backward]
    buf277 = aten.max_pool2d_with_indices_backward(buf276, mul_23, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_7)
    del buf276
    del getitem_7
    del mul_23
    buf278 = buf277
    del buf277
    buf279 = buf270; del buf270  # reuse
    buf280 = empty((64, ), device='cpu', dtype=torch.float32)
    buf281 = empty((64, ), device='cpu', dtype=torch.float32)
    buf282 = buf278; del buf278  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_46(c_void_p(buf282.data_ptr()), c_void_p(mul_606.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_472.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()))
    del buf280
    del convolution_2
    del mul_606
    del primals_5
    del squeeze_7
    del unsqueeze_472
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf283 = aten.convolution_backward(buf282, mul_15, primals_71, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf282
    del mul_15
    del primals_71
    buf284 = buf283[0]
    buf285 = buf283[1]
    del buf283
    buf286 = empty((32, ), device='cpu', dtype=torch.float32)
    buf287 = empty((32, ), device='cpu', dtype=torch.float32)
    buf288 = empty((32, ), device='cpu', dtype=torch.float32)
    buf289 = buf284; del buf284  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_47(c_void_p(buf289.data_ptr()), c_void_p(mul_618.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_484.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()))
    del buf287
    del convolution_1
    del mul_618
    del primals_3
    del squeeze_4
    del unsqueeze_484
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf290 = aten.convolution_backward(buf289, mul_7, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf289
    del mul_7
    del primals_70
    buf291 = buf290[0]
    buf292 = buf290[1]
    del buf290
    buf293 = empty((24, ), device='cpu', dtype=torch.float32)
    buf294 = empty((24, ), device='cpu', dtype=torch.float32)
    buf295 = empty((24, ), device='cpu', dtype=torch.float32)
    buf296 = buf291; del buf291  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_48(c_void_p(buf296.data_ptr()), c_void_p(mul_630.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_496.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()))
    del buf294
    del convolution
    del mul_630
    del primals_1
    del squeeze_1
    del unsqueeze_496
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf297 = aten.convolution_backward(buf296, primals_200, primals_69, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf296
    del primals_200
    del primals_69
    buf298 = buf297[1]
    return (buf295, buf293, buf288, buf286, buf281, buf279, buf271, buf269, buf264, buf261, buf251, buf243, buf245, buf243, buf237, buf235, buf230, buf227, buf218, buf215, buf210, buf208, buf203, buf200, buf190, buf182, buf184, buf182, buf176, buf174, buf169, buf166, buf157, buf154, buf149, buf147, buf142, buf139, buf129, buf121, buf123, buf121, buf115, buf113, reinterpret_tensor(buf104, (31, 16), (16, 1), 0), reinterpret_tensor(buf100, (31, 16), (16, 1), 0), buf93, buf91, buf87, buf84, buf79, buf77, reinterpret_tensor(buf68, (31, 16), (16, 1), 0), reinterpret_tensor(buf64, (31, 16), (16, 1), 0), buf56, buf54, buf49, buf39, buf42, buf39, buf34, buf32, reinterpret_tensor(buf23, (15, 16), (16, 1), 0), reinterpret_tensor(buf19, (15, 16), (16, 1), 0), buf12, buf10, buf5, buf3, buf298, buf292, buf285, buf275, buf268, buf260, buf255, buf249, buf241, buf234, buf226, buf221, buf214, buf207, buf199, buf194, buf188, buf180, buf173, buf165, buf160, buf153, buf146, buf138, buf133, buf127, buf119, buf112, buf90, buf83, buf76, buf53, buf46, buf38, buf31, buf9, reinterpret_tensor(buf1, (1000, 2048), (2048, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


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
    primals_96 = rand_strided((384, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((640, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((640, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cpu', dtype=torch.float32)
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
    view_17 = rand_strided((8192, 16), (16, 1), device='cpu', dtype=torch.float32)
    view_23 = rand_strided((8192, 16), (16, 1), device='cpu', dtype=torch.float32)
    bmm_1 = rand_strided((32, 256, 64), (16384, 64, 1), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    mul_186 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    mul_194 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    mul_202 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    view_41 = rand_strided((8192, 16), (16, 1), device='cpu', dtype=torch.float32)
    view_47 = rand_strided((8192, 16), (16, 1), device='cpu', dtype=torch.float32)
    view_57 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    avg_pool2d = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    mul_211 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    mul_226 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    mul_234 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    view_65 = rand_strided((2048, 16), (16, 1), device='cpu', dtype=torch.float32)
    view_71 = rand_strided((2048, 16), (16, 1), device='cpu', dtype=torch.float32)
    bmm_5 = rand_strided((32, 64, 128), (8192, 128, 1), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    mul_243 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    clone_48 = rand_strided((8, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_25 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    mul_253 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    unsqueeze_126 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_265 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    unsqueeze_138 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    permute_32 = rand_strided((32, 64, 64), (4096, 1, 64), device='cpu', dtype=torch.float32)
    permute_33 = rand_strided((32, 128, 64), (8192, 64, 1), device='cpu', dtype=torch.float32)
    alias_8 = rand_strided((32, 64, 64), (4096, 64, 1), device='cpu', dtype=torch.float32)
    permute_37 = rand_strided((15, 16), (16, 1), device='cpu', dtype=torch.float32)
    permute_43 = rand_strided((15, 16), (16, 1), device='cpu', dtype=torch.float32)
    permute_45 = rand_strided((32, 16, 64), (1024, 64, 1), device='cpu', dtype=torch.float32)
    permute_46 = rand_strided((32, 64, 16), (1024, 1, 64), device='cpu', dtype=torch.float32)
    mul_280 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    unsqueeze_150 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_292 = rand_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    unsqueeze_162 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_174 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_313 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cpu', dtype=torch.float32)
    unsqueeze_186 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    permute_53 = rand_strided((32, 256, 256), (65536, 1, 256), device='cpu', dtype=torch.float32)
    permute_54 = rand_strided((32, 128, 256), (32768, 256, 1), device='cpu', dtype=torch.float32)
    alias_9 = rand_strided((32, 256, 256), (65536, 256, 1), device='cpu', dtype=torch.float32)
    permute_58 = rand_strided((31, 16), (16, 1), device='cpu', dtype=torch.float32)
    permute_64 = rand_strided((31, 16), (16, 1), device='cpu', dtype=torch.float32)
    permute_66 = rand_strided((32, 16, 256), (4096, 256, 1), device='cpu', dtype=torch.float32)
    permute_67 = rand_strided((32, 256, 16), (4096, 1, 256), device='cpu', dtype=torch.float32)
    mul_328 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    unsqueeze_198 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_340 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cpu', dtype=torch.float32)
    unsqueeze_210 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_352 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cpu', dtype=torch.float32)
    unsqueeze_222 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    permute_74 = rand_strided((32, 256, 256), (65536, 1, 256), device='cpu', dtype=torch.float32)
    permute_75 = rand_strided((32, 64, 256), (16384, 256, 1), device='cpu', dtype=torch.float32)
    alias_10 = rand_strided((32, 256, 256), (65536, 256, 1), device='cpu', dtype=torch.float32)
    permute_79 = rand_strided((31, 16), (16, 1), device='cpu', dtype=torch.float32)
    permute_85 = rand_strided((31, 16), (16, 1), device='cpu', dtype=torch.float32)
    permute_87 = rand_strided((32, 16, 256), (4096, 256, 1), device='cpu', dtype=torch.float32)
    permute_88 = rand_strided((32, 256, 16), (4096, 1, 256), device='cpu', dtype=torch.float32)
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
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_47, primals_49, primals_51, primals_55, primals_57, primals_59, primals_61, primals_65, primals_67, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_200, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, getitem_6, getitem_7, convolution_3, squeeze_10, mul_31, convolution_4, squeeze_13, add_24, view, convolution_5, mul_40, convolution_6, squeeze_16, convolution_7, squeeze_19, mul_55, convolution_8, squeeze_22, mul_63, convolution_9, squeeze_25, add_45, view_2, convolution_10, mul_72, convolution_11, squeeze_28, mul_80, convolution_12, squeeze_31, mul_88, convolution_13, squeeze_34, add_61, view_4, convolution_14, mul_97, convolution_15, squeeze_37, convolution_16, squeeze_40, mul_112, convolution_17, squeeze_43, mul_120, convolution_18, squeeze_46, add_82, view_6, convolution_19, mul_129, convolution_20, squeeze_49, mul_137, convolution_21, squeeze_52, mul_145, convolution_22, squeeze_55, add_98, view_8, convolution_23, mul_154, convolution_24, squeeze_58, convolution_25, squeeze_61, mul_169, convolution_26, squeeze_64, mul_177, view_17, view_23, bmm_1, squeeze_67, mul_186, convolution_28, squeeze_70, mul_194, convolution_29, squeeze_73, mul_202, view_41, view_47, view_57, avg_pool2d, squeeze_76, mul_211, convolution_31, squeeze_79, convolution_32, squeeze_82, mul_226, convolution_33, squeeze_85, mul_234, view_65, view_71, bmm_5, squeeze_88, mul_243, convolution_35, squeeze_91, clone_48, permute_25, mul_253, unsqueeze_126, mul_265, unsqueeze_138, permute_32, permute_33, alias_8, permute_37, permute_43, permute_45, permute_46, mul_280, unsqueeze_150, mul_292, unsqueeze_162, unsqueeze_174, mul_313, unsqueeze_186, permute_53, permute_54, alias_9, permute_58, permute_64, permute_66, permute_67, mul_328, unsqueeze_198, mul_340, unsqueeze_210, mul_352, unsqueeze_222, permute_74, permute_75, alias_10, permute_79, permute_85, permute_87, permute_88, mul_367, unsqueeze_234, mul_379, unsqueeze_246, unsqueeze_258, unsqueeze_272, mul_416, unsqueeze_284, mul_428, unsqueeze_296, unsqueeze_310, mul_456, unsqueeze_322, mul_468, unsqueeze_334, unsqueeze_346, unsqueeze_360, mul_505, unsqueeze_372, mul_517, unsqueeze_384, unsqueeze_398, mul_545, unsqueeze_410, mul_557, unsqueeze_422, unsqueeze_434, unsqueeze_448, mul_594, unsqueeze_460, mul_606, unsqueeze_472, mul_618, unsqueeze_484, mul_630, unsqueeze_496, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('eca_botnext26ts_256', benchmark_compiled_module)
