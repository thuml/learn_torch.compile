
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


cpp_fused_as_strided_scatter_clone_native_layer_norm_backward_squeeze_sum_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
    auto in_ptr1 = in_out_ptr0;
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp4;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr4[static_cast<long>(x0)];
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = out_ptr1[static_cast<long>(x0)];
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                auto tmp11 = out_ptr2[static_cast<long>(x0)];
                auto tmp3 = tmp1 * tmp2;
                auto tmp4 = static_cast<float>(1024.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 * tmp5;
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 - tmp8;
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp10 * tmp12;
                auto tmp14 = tmp9 - tmp13;
                auto tmp15 = at::vec::Vectorized<float>(tmp0);
                auto tmp16 = tmp15 * tmp14;
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr5 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(out_ptr5 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x2 + (1024L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(49.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        auto tmp5 = tmp3 * tmp4;
                        tmp5.store(out_ptr6 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (1024L*x2) + (7168L*x1) + (50176L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (1024L*x2) + (7168L*x1) + (50176L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (7L*x2) + (49L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (7L*x2) + (49L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (7L*x1) + (49L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (7L*x2) + (49L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (1024L*x2) + (7168L*x1) + (50176L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (7L*x1) + (49L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (1024L*x2) + (7168L*x1) + (50176L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (7L*x2) + (49L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (7L*x1) + (49L*x0))];
                            auto tmp1 = static_cast<float>(1024.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (1024L*x1) + (7168L*x2) + (50176L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_3 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(49.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 * tmp6;
                        tmp7.store(out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (1024L*x2) + (7168L*x1) + (50176L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (1024L*x2) + (7168L*x1) + (50176L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (7L*x2) + (49L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (7L*x2) + (49L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (7L*x1) + (49L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (7L*x2) + (49L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (1024L*x2) + (7168L*x1) + (50176L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (7L*x1) + (49L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (1024L*x2) + (7168L*x1) + (50176L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (7L*x2) + (49L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (7L*x1) + (49L*x0))];
                            auto tmp1 = static_cast<float>(1024.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (1024L*x1) + (7168L*x2) + (50176L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_sum_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x2) + (50176L*x1)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x2) + (50176L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x2) + (50176L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (1024L*x2) + (50176L*x1)));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x2) + (50176L*x1)));
                            auto tmp1 = static_cast<float>(49.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp7 = tmp3 + tmp6;
                            auto tmp9 = tmp7 * tmp8;
                            auto tmp11 = tmp7 + tmp10;
                            auto tmp13 = tmp11 * tmp12;
                            tmp_acc0_vec = tmp_acc0_vec + tmp5;
                            tmp_acc1_vec = tmp_acc1_vec + tmp9;
                            tmp_acc2_vec = tmp_acc2_vec + tmp13;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(7L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x3) + (7168L*x2) + (50176L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x3) + (7168L*x2) + (50176L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (7L*x3) + (49L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (7L*x3) + (49L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(7L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x3) + (7168L*x2) + (50176L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x3) + (7168L*x2) + (50176L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (7L*x3) + (49L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (7L*x3) + (49L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(49.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_sum_13 = async_compile.cpp('''
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
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (1024L*x2) + (7168L*x1) + (50176L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (1024L*x2) + (7168L*x1) + (50176L*x0)));
                                auto tmp4 = in_ptr4[static_cast<long>(x1 + (7L*x2) + (49L*x0))];
                                auto tmp7 = in_ptr5[static_cast<long>(x1 + (7L*x2) + (49L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr2[static_cast<long>(x2 + (7L*x1) + (49L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(7L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x3) + (7168L*x2) + (50176L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x3) + (7168L*x2) + (50176L*x1)));
                                auto tmp2 = in_ptr4[static_cast<long>(x2 + (7L*x3) + (49L*x1))];
                                auto tmp5 = in_ptr5[static_cast<long>(x2 + (7L*x3) + (49L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr5[static_cast<long>(x1 + (7L*x2) + (49L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (1024L*x2) + (7168L*x1) + (50176L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (7L*x1) + (49L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (1024L*x2) + (7168L*x1) + (50176L*x0)));
                            auto tmp12 = in_ptr4[static_cast<long>(x1 + (7L*x2) + (49L*x0))];
                            auto tmp17 = out_ptr2[static_cast<long>(x2 + (7L*x1) + (49L*x0))];
                            auto tmp1 = static_cast<float>(1024.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr5 + static_cast<long>(x3 + (1024L*x1) + (7168L*x2) + (50176L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        float tmp10[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (50176L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (50176L*x0)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1024L*x1) + (1024L*x1_inner) + (50176L*x0)));
                            auto tmp1 = static_cast<float>(49.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = tmp5 + tmp6;
                            auto tmp9 = tmp7 + tmp8;
                            tmp9.store(tmp10 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp10, 8, out_ptr0 + static_cast<long>(x1 + (49L*x2) + (50176L*x0)), static_cast<long>(49L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1024L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x2 + (1024L*x1) + (50176L*x0))];
                        auto tmp5 = in_ptr2[static_cast<long>(x2 + (1024L*x1) + (50176L*x0))];
                        auto tmp7 = in_ptr3[static_cast<long>(x2 + (1024L*x1) + (50176L*x0))];
                        auto tmp1 = static_cast<float>(49.0);
                        auto tmp2 = tmp0 / tmp1;
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        out_ptr0[static_cast<long>(x1 + (49L*x2) + (50176L*x0))] = tmp8;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_native_layer_norm_backward_view_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (7168L*x2) + (100352L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = tmp2 * tmp3;
                                tmp_acc0_vec = tmp_acc0_vec + tmp4;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr2[static_cast<long>(x2 + (14L*x0) + (14L*x0_inner) + (7168L*x3) + (100352L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (7168L*x2) + (100352L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp11 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = static_cast<float>(512.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 - tmp8;
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 * tmp12;
                            auto tmp14 = tmp9 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp15 * tmp14;
                            tmp16.store(in_out_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr4 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 * tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr4[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr4[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp5[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 * tmp3;
                            tmp4.store(tmp5 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp5, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp4;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 * tmp5;
                            tmp6.store(tmp7 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp7, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp6;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp4 + tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp8 + tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
    }
}
''')


cpp_fused_sum_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp9[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 * tmp7;
                            tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp7 = in_ptr4[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_sum_33 = async_compile.cpp('''
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
                       float* out_ptr5)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr5[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr2[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr5[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr5[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr2[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr5 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 * tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp5[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 * tmp3;
                            tmp4.store(tmp5 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp5, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp4;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 * tmp5;
                            tmp6.store(tmp7 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp7, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp6;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp4 + tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp8 + tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
    }
}
''')


cpp_fused_sum_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp9[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 * tmp7;
                            tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp7 = in_ptr4[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_sum_52 = async_compile.cpp('''
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
                       float* out_ptr5)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr5[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr2[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr5[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr5[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr2[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr5 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 * tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp5[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 * tmp3;
                            tmp4.store(tmp5 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp5, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp4;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 * tmp5;
                            tmp6.store(tmp7 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp7, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp6;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp4 + tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp8 + tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
    }
}
''')


cpp_fused_sum_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp9[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 * tmp7;
                            tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp7 = in_ptr4[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_sum_71 = async_compile.cpp('''
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
                       float* out_ptr5)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr5[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr2[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr5[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr5[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr2[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr5 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 * tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_75 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp5[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 * tmp3;
                            tmp4.store(tmp5 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp5, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp4;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 * tmp5;
                            tmp6.store(tmp7 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp7, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp6;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp4 + tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp8 + tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
    }
}
''')


cpp_fused_sum_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp9[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 * tmp7;
                            tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp7 = in_ptr4[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_sum_90 = async_compile.cpp('''
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
                       float* out_ptr5)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr5[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr2[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr5[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr5[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr2[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr5 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 * tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_94 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp5[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 * tmp3;
                            tmp4.store(tmp5 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp5, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp4;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 * tmp5;
                            tmp6.store(tmp7 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp7, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp6;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp4 + tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp8 + tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
    }
}
''')


cpp_fused_sum_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp9[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 * tmp7;
                            tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp7 = in_ptr4[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_sum_109 = async_compile.cpp('''
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
                       float* out_ptr5)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr5[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr2[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr5[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr5[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr2[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr5 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 * tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_113 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp5[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 * tmp3;
                            tmp4.store(tmp5 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp5, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp4;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_115 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 * tmp5;
                            tmp6.store(tmp7 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp7, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp6;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp4 + tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp8 + tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
    }
}
''')


cpp_fused_sum_120 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_122 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_124 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_125 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_126 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp9[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 * tmp7;
                            tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp7 = in_ptr4[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp8;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_127 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_sum_128 = async_compile.cpp('''
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
                       float* out_ptr5)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr5[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr2[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr5[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr5[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr2[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr5 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_129 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 * tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_130 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_131 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_132 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp5[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 * tmp3;
                            tmp4.store(tmp5 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp5, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp4;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_133 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_134 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_135 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (512L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp4 + tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                        tmp_acc2_vec = tmp_acc2_vec + tmp10;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_136 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_137 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_138 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_139 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_140 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 * tmp5;
                            tmp6.store(tmp7 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp7, 8, out_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        out_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))] = tmp6;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((196L*x1) + (196L*x1_inner) + (100352L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(x0) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_141 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_sum_142 = async_compile.cpp('''
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
                       float* out_ptr5)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2048L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp4 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp7 = in_ptr5[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr2[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (512L*x3) + (7168L*x2) + (100352L*x1)));
                                auto tmp2 = in_ptr4[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp5 = in_ptr5[static_cast<long>(x2 + (14L*x3) + (196L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr5[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp12 = in_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))];
                            auto tmp17 = out_ptr2[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr5 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_143 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp6.store(out_ptr0 + static_cast<long>(x3 + (512L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_native_layer_norm_backward_view_144 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (28L*x3) + (28L*x3_inner) + (7168L*x2) + (200704L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = tmp2 * tmp3;
                                tmp_acc0_vec = tmp_acc0_vec + tmp4;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(28L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x3) + (7168L*x2) + (200704L*x1)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr2[static_cast<long>(x2 + (28L*x0) + (28L*x0_inner) + (7168L*x3) + (200704L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr3[static_cast<long>(x1 + (28L*x2) + (784L*x0))];
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (28L*x3) + (28L*x3_inner) + (7168L*x2) + (200704L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp11 = out_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = static_cast<float>(256.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 - tmp8;
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 * tmp12;
                            auto tmp14 = tmp9 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp15 * tmp14;
                            tmp16.store(in_out_ptr0 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (256L*x2) + (200704L*x0)), static_cast<long>(256L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr4 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr4[static_cast<long>((784L*x1) + (784L*x1_inner) + (200704L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(x0) % static_cast<long>(784L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_145 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_146 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (28L*x2) + (784L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (28L*x2) + (784L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (28L*x2) + (784L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (28L*x2) + (784L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp1 = static_cast<float>(256.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (256L*x1) + (7168L*x2) + (200704L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_147 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp5[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (200704L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (200704L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 * tmp3;
                            tmp4.store(tmp5 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp5, 8, out_ptr0 + static_cast<long>(x1 + (784L*x2) + (200704L*x0)), static_cast<long>(784L));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((784L*x1) + (784L*x1_inner) + (200704L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(x0) % static_cast<long>(784L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_148 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_149 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (28L*x2) + (784L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (28L*x2) + (784L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (28L*x2) + (784L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (28L*x2) + (784L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp1 = static_cast<float>(256.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (256L*x1) + (7168L*x2) + (200704L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_sum_150 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp4 + tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                        tmp_acc2_vec = tmp_acc2_vec + tmp10;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_151 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_152 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(28L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x3) + (7168L*x2) + (200704L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x3) + (7168L*x2) + (200704L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (28L*x3) + (784L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (28L*x3) + (784L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_153 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_154 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(28L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x3) + (7168L*x2) + (200704L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x3) + (7168L*x2) + (200704L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (28L*x3) + (784L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (28L*x3) + (784L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_155 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (200704L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (200704L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (200704L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 * tmp5;
                            tmp6.store(tmp7 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp7, 8, out_ptr0 + static_cast<long>(x1 + (784L*x2) + (200704L*x0)), static_cast<long>(784L));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((784L*x1) + (784L*x1_inner) + (200704L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(x0) % static_cast<long>(784L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_156 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_sum_157 = async_compile.cpp('''
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
                       float* out_ptr5)
{
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1024L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                                auto tmp4 = in_ptr4[static_cast<long>(x1 + (28L*x2) + (784L*x0))];
                                auto tmp7 = in_ptr5[static_cast<long>(x1 + (28L*x2) + (784L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr2[static_cast<long>(x2 + (28L*x1) + (784L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(28L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x3) + (7168L*x2) + (200704L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x3) + (7168L*x2) + (200704L*x1)));
                                auto tmp2 = in_ptr4[static_cast<long>(x2 + (28L*x3) + (784L*x1))];
                                auto tmp5 = in_ptr5[static_cast<long>(x2 + (28L*x3) + (784L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr5[static_cast<long>(x1 + (28L*x2) + (784L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                            auto tmp12 = in_ptr4[static_cast<long>(x1 + (28L*x2) + (784L*x0))];
                            auto tmp17 = out_ptr2[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp1 = static_cast<float>(256.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr5 + static_cast<long>(x3 + (256L*x1) + (7168L*x2) + (200704L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_158 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp6.store(out_ptr0 + static_cast<long>(x3 + (256L*x1) + (7168L*x2) + (200704L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_native_layer_norm_backward_view_159 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (56L*x3) + (56L*x3_inner) + (7168L*x2) + (401408L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = tmp2 * tmp3;
                                tmp_acc0_vec = tmp_acc0_vec + tmp4;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (56L*x1) + (3136L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(56L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x3) + (7168L*x2) + (401408L*x1)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr2[static_cast<long>(x2 + (56L*x0) + (56L*x0_inner) + (7168L*x3) + (401408L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr3[static_cast<long>(x1 + (56L*x2) + (3136L*x0))];
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (56L*x1) + (3136L*x0))];
                            auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (56L*x3) + (56L*x3_inner) + (7168L*x2) + (401408L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp11 = out_ptr1[static_cast<long>(x2 + (56L*x1) + (3136L*x0))];
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp4 = static_cast<float>(128.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 - tmp8;
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 * tmp12;
                            auto tmp14 = tmp9 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp15 * tmp14;
                            tmp16.store(in_out_ptr0 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (128L*x2) + (401408L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr4 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (401408L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr4[static_cast<long>((3136L*x1) + (3136L*x1_inner) + (401408L*(c10::div_floor_integer(x0, 3136L))) + (static_cast<long>(x0) % static_cast<long>(3136L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr5 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_160 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12845056L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_161 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (56L*x2) + (3136L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (56L*x2) + (3136L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (56L*x1) + (3136L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (56L*x2) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (56L*x1) + (3136L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (56L*x2) + (3136L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (56L*x1) + (3136L*x0))];
                            auto tmp1 = static_cast<float>(128.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (128L*x1) + (7168L*x2) + (401408L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_162 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp5[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (401408L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (401408L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 * tmp3;
                            tmp4.store(tmp5 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp5, 8, out_ptr0 + static_cast<long>(x1 + (3136L*x2) + (401408L*x0)), static_cast<long>(3136L));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((3136L*x1) + (3136L*x1_inner) + (401408L*(c10::div_floor_integer(x0, 3136L))) + (static_cast<long>(x0) % static_cast<long>(3136L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_163 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12845056L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_164 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (56L*x2) + (3136L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (56L*x2) + (3136L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (56L*x1) + (3136L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (56L*x2) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (56L*x1) + (3136L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (56L*x2) + (3136L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (56L*x1) + (3136L*x0))];
                            auto tmp1 = static_cast<float>(128.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (128L*x1) + (7168L*x2) + (401408L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_view_165 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (401408L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (401408L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (401408L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 * tmp5;
                            tmp6.store(tmp7 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp7, 8, out_ptr0 + static_cast<long>(x1 + (3136L*x2) + (401408L*x0)), static_cast<long>(3136L));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((3136L*x1) + (3136L*x1_inner) + (401408L*(c10::div_floor_integer(x0, 3136L))) + (static_cast<long>(x0) % static_cast<long>(3136L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_166 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12845056L); x0+=static_cast<long>(8L))
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


cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_167 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x1 + (56L*x2) + (3136L*x0))];
                                auto tmp7 = in_ptr4[static_cast<long>(x1 + (56L*x2) + (3136L*x0))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 - tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 * tmp8;
                                auto tmp10 = tmp2 * tmp9;
                                tmp_acc0_vec = tmp_acc0_vec + tmp10;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr1[static_cast<long>(x2 + (56L*x1) + (3136L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x1 + (56L*x2) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (56L*x1) + (3136L*x0))];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                            auto tmp12 = in_ptr3[static_cast<long>(x1 + (56L*x2) + (3136L*x0))];
                            auto tmp17 = out_ptr1[static_cast<long>(x2 + (56L*x1) + (3136L*x0))];
                            auto tmp1 = static_cast<float>(128.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp6 = at::vec::Vectorized<float>(tmp1);
                            auto tmp7 = tmp5 * tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 - tmp13;
                            auto tmp15 = at::vec::Vectorized<float>(tmp0);
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 * tmp18;
                            auto tmp20 = tmp10 - tmp19;
                            auto tmp21 = at::vec::Vectorized<float>(tmp2);
                            auto tmp22 = tmp21 * tmp20;
                            tmp22.store(out_ptr2 + static_cast<long>(x3 + (128L*x1) + (7168L*x2) + (401408L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_backward_sum_168 = async_compile.cpp('''
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp4 + tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        auto tmp12 = tmp8 + tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                        tmp_acc2_vec = tmp_acc2_vec + tmp10;
                        tmp_acc3_vec = tmp_acc3_vec + tmp12;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_169 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_170 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(56L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x3) + (7168L*x2) + (401408L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x3) + (7168L*x2) + (401408L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (56L*x3) + (3136L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (56L*x3) + (3136L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_171 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_172 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(56L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x3) + (7168L*x2) + (401408L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x3) + (7168L*x2) + (401408L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (56L*x3) + (3136L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (56L*x3) + (3136L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_173 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_sum_174 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(56L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x3) + (7168L*x2) + (401408L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x3) + (7168L*x2) + (401408L*x1)));
                                auto tmp2 = in_ptr3[static_cast<long>(x2 + (56L*x3) + (3136L*x1))];
                                auto tmp5 = in_ptr4[static_cast<long>(x2 + (56L*x3) + (3136L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 * tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_layer_norm_backward_175 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp8.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr5[static_cast<long>(x1 + (56L*x3) + (56L*x3_inner) + (7168L*x2) + (401408L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr2[static_cast<long>(x2 + (56L*x1) + (3136L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(56L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x3) + (7168L*x2) + (401408L*x1)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x3) + (7168L*x2) + (401408L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x3) + (7168L*x2) + (401408L*x1)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x3) + (7168L*x2) + (401408L*x1)));
                                auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr5[static_cast<long>(x2 + (56L*x0) + (56L*x0_inner) + (7168L*x3) + (401408L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 + tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                auto tmp6 = tmp4 + tmp5;
                                auto tmp8 = tmp6 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp8;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr6[static_cast<long>(x1 + (56L*x2) + (3136L*x0))];
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                            auto tmp5 = out_ptr1[static_cast<long>(x2 + (56L*x1) + (3136L*x0))];
                            auto tmp8 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr5[static_cast<long>(x1 + (56L*x3) + (56L*x3_inner) + (7168L*x2) + (401408L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp9 = out_ptr2[static_cast<long>(x2 + (56L*x1) + (3136L*x0))];
                            auto tmp2 = static_cast<float>(128.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 - tmp6;
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 * tmp10;
                            auto tmp12 = tmp7 - tmp11;
                            auto tmp13 = at::vec::Vectorized<float>(tmp0);
                            auto tmp14 = tmp13 * tmp12;
                            tmp14.store(out_ptr4 + static_cast<long>(x3 + (128L*x1) + (7168L*x2) + (401408L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_110, primals_111, primals_113, primals_114, primals_116, primals_117, primals_119, primals_121, primals_127, primals_133, primals_139, primals_141, primals_147, primals_153, primals_159, primals_161, primals_167, primals_173, primals_179, primals_185, primals_191, primals_197, primals_203, primals_209, primals_215, primals_221, primals_227, primals_233, primals_239, primals_245, primals_251, primals_257, primals_263, primals_269, primals_275, primals_281, primals_287, primals_293, primals_299, primals_305, primals_311, primals_317, primals_323, primals_325, primals_331, primals_337, primals_345, mul, permute_1, convolution_1, getitem_3, rsqrt_1, view, addmm, view_2, addmm_1, add_5, convolution_2, getitem_5, rsqrt_2, view_5, addmm_2, view_7, addmm_3, add_9, convolution_3, getitem_7, rsqrt_3, view_10, addmm_4, view_12, addmm_5, mul_20, permute_15, convolution_4, convolution_5, getitem_11, rsqrt_5, view_15, addmm_6, view_17, addmm_7, add_19, convolution_6, getitem_13, rsqrt_6, view_20, addmm_8, view_22, addmm_9, add_23, convolution_7, getitem_15, rsqrt_7, view_25, addmm_10, view_27, addmm_11, mul_40, permute_29, convolution_8, convolution_9, getitem_19, rsqrt_9, view_30, addmm_12, view_32, addmm_13, add_33, convolution_10, getitem_21, rsqrt_10, view_35, addmm_14, view_37, addmm_15, add_37, convolution_11, getitem_23, rsqrt_11, view_40, addmm_16, view_42, addmm_17, add_41, convolution_12, getitem_25, rsqrt_12, view_45, addmm_18, view_47, addmm_19, add_45, convolution_13, getitem_27, rsqrt_13, view_50, addmm_20, view_52, addmm_21, add_49, convolution_14, getitem_29, rsqrt_14, view_55, addmm_22, view_57, addmm_23, add_53, convolution_15, getitem_31, rsqrt_15, view_60, addmm_24, view_62, addmm_25, add_57, convolution_16, getitem_33, rsqrt_16, view_65, addmm_26, view_67, addmm_27, add_61, convolution_17, getitem_35, rsqrt_17, view_70, addmm_28, view_72, addmm_29, add_65, convolution_18, getitem_37, rsqrt_18, view_75, addmm_30, view_77, addmm_31, add_69, convolution_19, getitem_39, rsqrt_19, view_80, addmm_32, view_82, addmm_33, add_73, convolution_20, getitem_41, rsqrt_20, view_85, addmm_34, view_87, addmm_35, add_77, convolution_21, getitem_43, rsqrt_21, view_90, addmm_36, view_92, addmm_37, add_81, convolution_22, getitem_45, rsqrt_22, view_95, addmm_38, view_97, addmm_39, add_85, convolution_23, getitem_47, rsqrt_23, view_100, addmm_40, view_102, addmm_41, add_89, convolution_24, getitem_49, rsqrt_24, view_105, addmm_42, view_107, addmm_43, add_93, convolution_25, getitem_51, rsqrt_25, view_110, addmm_44, view_112, addmm_45, add_97, convolution_26, getitem_53, rsqrt_26, view_115, addmm_46, view_117, addmm_47, add_101, convolution_27, getitem_55, rsqrt_27, view_120, addmm_48, view_122, addmm_49, add_105, convolution_28, getitem_57, rsqrt_28, view_125, addmm_50, view_127, addmm_51, add_109, convolution_29, getitem_59, rsqrt_29, view_130, addmm_52, view_132, addmm_53, add_113, convolution_30, getitem_61, rsqrt_30, view_135, addmm_54, view_137, addmm_55, add_117, convolution_31, getitem_63, rsqrt_31, view_140, addmm_56, view_142, addmm_57, add_121, convolution_32, getitem_65, rsqrt_32, view_145, addmm_58, view_147, addmm_59, add_125, convolution_33, getitem_67, rsqrt_33, view_150, addmm_60, view_152, addmm_61, add_129, convolution_34, getitem_69, rsqrt_34, view_155, addmm_62, view_157, addmm_63, add_133, convolution_35, getitem_71, rsqrt_35, view_160, addmm_64, view_162, addmm_65, mul_204, permute_139, convolution_36, convolution_37, getitem_75, rsqrt_37, view_165, addmm_66, view_167, addmm_67, add_143, convolution_38, getitem_77, rsqrt_38, view_170, addmm_68, view_172, addmm_69, add_147, convolution_39, getitem_79, rsqrt_39, view_175, addmm_70, view_177, addmm_71, mul_224, clone_73, permute_155, div, permute_162, permute_166, permute_172, permute_176, permute_182, permute_186, div_5, permute_194, permute_198, permute_204, permute_208, permute_214, permute_218, permute_224, permute_228, permute_234, permute_238, permute_244, permute_248, permute_254, permute_258, permute_264, permute_268, permute_274, permute_278, permute_284, permute_288, permute_294, permute_298, permute_304, permute_308, permute_314, permute_318, permute_324, permute_328, permute_334, permute_338, permute_344, permute_348, permute_354, permute_358, permute_364, permute_368, permute_374, permute_378, permute_384, permute_388, permute_394, permute_398, permute_404, permute_408, permute_414, permute_418, permute_424, permute_428, permute_434, permute_438, permute_444, permute_448, permute_454, permute_458, div_33, permute_466, permute_470, permute_476, permute_480, permute_486, permute_490, div_37, permute_498, permute_502, permute_508, permute_512, permute_518, permute_522, div_41, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (128, ), (1, ))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_28, (512, ), (1, ))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (512, ), (1, ))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_36, (512, ), (1, ))
    assert_size_stride(primals_37, (512, ), (1, ))
    assert_size_stride(primals_39, (512, ), (1, ))
    assert_size_stride(primals_40, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (512, ), (1, ))
    assert_size_stride(primals_45, (512, ), (1, ))
    assert_size_stride(primals_46, (512, ), (1, ))
    assert_size_stride(primals_48, (512, ), (1, ))
    assert_size_stride(primals_49, (512, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_52, (512, ), (1, ))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (512, ), (1, ))
    assert_size_stride(primals_57, (512, ), (1, ))
    assert_size_stride(primals_58, (512, ), (1, ))
    assert_size_stride(primals_60, (512, ), (1, ))
    assert_size_stride(primals_61, (512, ), (1, ))
    assert_size_stride(primals_63, (512, ), (1, ))
    assert_size_stride(primals_64, (512, ), (1, ))
    assert_size_stride(primals_66, (512, ), (1, ))
    assert_size_stride(primals_67, (512, ), (1, ))
    assert_size_stride(primals_69, (512, ), (1, ))
    assert_size_stride(primals_70, (512, ), (1, ))
    assert_size_stride(primals_72, (512, ), (1, ))
    assert_size_stride(primals_73, (512, ), (1, ))
    assert_size_stride(primals_75, (512, ), (1, ))
    assert_size_stride(primals_76, (512, ), (1, ))
    assert_size_stride(primals_78, (512, ), (1, ))
    assert_size_stride(primals_79, (512, ), (1, ))
    assert_size_stride(primals_81, (512, ), (1, ))
    assert_size_stride(primals_82, (512, ), (1, ))
    assert_size_stride(primals_84, (512, ), (1, ))
    assert_size_stride(primals_85, (512, ), (1, ))
    assert_size_stride(primals_87, (512, ), (1, ))
    assert_size_stride(primals_88, (512, ), (1, ))
    assert_size_stride(primals_90, (512, ), (1, ))
    assert_size_stride(primals_91, (512, ), (1, ))
    assert_size_stride(primals_93, (512, ), (1, ))
    assert_size_stride(primals_94, (512, ), (1, ))
    assert_size_stride(primals_96, (512, ), (1, ))
    assert_size_stride(primals_97, (512, ), (1, ))
    assert_size_stride(primals_99, (512, ), (1, ))
    assert_size_stride(primals_100, (512, ), (1, ))
    assert_size_stride(primals_102, (512, ), (1, ))
    assert_size_stride(primals_103, (512, ), (1, ))
    assert_size_stride(primals_105, (512, ), (1, ))
    assert_size_stride(primals_106, (512, ), (1, ))
    assert_size_stride(primals_108, (1024, ), (1, ))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_111, (1024, ), (1, ))
    assert_size_stride(primals_113, (1024, ), (1, ))
    assert_size_stride(primals_114, (1024, ), (1, ))
    assert_size_stride(primals_116, (1024, ), (1, ))
    assert_size_stride(primals_117, (1024, ), (1, ))
    assert_size_stride(primals_119, (128, 3, 4, 4), (48, 1, 12, 3))
    assert_size_stride(primals_121, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_127, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_133, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_139, (256, 128, 2, 2), (512, 1, 256, 128))
    assert_size_stride(primals_141, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_147, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_153, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_159, (512, 256, 2, 2), (1024, 1, 512, 256))
    assert_size_stride(primals_161, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_167, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_173, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_179, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_185, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_191, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_197, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_203, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_209, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_215, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_221, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_227, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_233, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_239, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_245, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_251, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_257, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_263, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_269, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_275, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_281, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_287, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_293, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_299, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_305, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_311, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_317, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_323, (1024, 512, 2, 2), (2048, 1, 1024, 512))
    assert_size_stride(primals_325, (1024, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_331, (1024, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_337, (1024, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_345, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(mul, (8, 56, 56, 128), (401408, 1, 7168, 56))
    assert_size_stride(permute_1, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_1, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(getitem_3, (8, 56, 56, 1), (3136, 1, 56, 56))
    assert_size_stride(rsqrt_1, (8, 56, 56, 1), (3136, 1, 56, 56))
    assert_size_stride(view, (25088, 128), (128, 1))
    assert_size_stride(addmm, (25088, 512), (512, 1))
    assert_size_stride(view_2, (25088, 512), (512, 1))
    assert_size_stride(addmm_1, (25088, 128), (128, 1))
    assert_size_stride(add_5, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_2, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(getitem_5, (8, 56, 56, 1), (3136, 1, 56, 56))
    assert_size_stride(rsqrt_2, (8, 56, 56, 1), (3136, 1, 56, 56))
    assert_size_stride(view_5, (25088, 128), (128, 1))
    assert_size_stride(addmm_2, (25088, 512), (512, 1))
    assert_size_stride(view_7, (25088, 512), (512, 1))
    assert_size_stride(addmm_3, (25088, 128), (128, 1))
    assert_size_stride(add_9, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_3, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(getitem_7, (8, 56, 56, 1), (3136, 1, 56, 56))
    assert_size_stride(rsqrt_3, (8, 56, 56, 1), (3136, 1, 56, 56))
    assert_size_stride(view_10, (25088, 128), (128, 1))
    assert_size_stride(addmm_4, (25088, 512), (512, 1))
    assert_size_stride(view_12, (25088, 512), (512, 1))
    assert_size_stride(addmm_5, (25088, 128), (128, 1))
    assert_size_stride(mul_20, (8, 56, 56, 128), (401408, 1, 7168, 56))
    assert_size_stride(permute_15, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convolution_4, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_5, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(getitem_11, (8, 28, 28, 1), (784, 1, 28, 28))
    assert_size_stride(rsqrt_5, (8, 28, 28, 1), (784, 1, 28, 28))
    assert_size_stride(view_15, (6272, 256), (256, 1))
    assert_size_stride(addmm_6, (6272, 1024), (1024, 1))
    assert_size_stride(view_17, (6272, 1024), (1024, 1))
    assert_size_stride(addmm_7, (6272, 256), (256, 1))
    assert_size_stride(add_19, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_6, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(getitem_13, (8, 28, 28, 1), (784, 1, 28, 28))
    assert_size_stride(rsqrt_6, (8, 28, 28, 1), (784, 1, 28, 28))
    assert_size_stride(view_20, (6272, 256), (256, 1))
    assert_size_stride(addmm_8, (6272, 1024), (1024, 1))
    assert_size_stride(view_22, (6272, 1024), (1024, 1))
    assert_size_stride(addmm_9, (6272, 256), (256, 1))
    assert_size_stride(add_23, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_7, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(getitem_15, (8, 28, 28, 1), (784, 1, 28, 28))
    assert_size_stride(rsqrt_7, (8, 28, 28, 1), (784, 1, 28, 28))
    assert_size_stride(view_25, (6272, 256), (256, 1))
    assert_size_stride(addmm_10, (6272, 1024), (1024, 1))
    assert_size_stride(view_27, (6272, 1024), (1024, 1))
    assert_size_stride(addmm_11, (6272, 256), (256, 1))
    assert_size_stride(mul_40, (8, 28, 28, 256), (200704, 1, 7168, 28))
    assert_size_stride(permute_29, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convolution_8, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_9, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_19, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_9, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_30, (1568, 512), (512, 1))
    assert_size_stride(addmm_12, (1568, 2048), (2048, 1))
    assert_size_stride(view_32, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_13, (1568, 512), (512, 1))
    assert_size_stride(add_33, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_10, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_21, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_10, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_35, (1568, 512), (512, 1))
    assert_size_stride(addmm_14, (1568, 2048), (2048, 1))
    assert_size_stride(view_37, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_15, (1568, 512), (512, 1))
    assert_size_stride(add_37, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_11, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_23, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_11, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_40, (1568, 512), (512, 1))
    assert_size_stride(addmm_16, (1568, 2048), (2048, 1))
    assert_size_stride(view_42, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_17, (1568, 512), (512, 1))
    assert_size_stride(add_41, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_12, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_25, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_12, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_45, (1568, 512), (512, 1))
    assert_size_stride(addmm_18, (1568, 2048), (2048, 1))
    assert_size_stride(view_47, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_19, (1568, 512), (512, 1))
    assert_size_stride(add_45, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_13, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_27, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_13, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_50, (1568, 512), (512, 1))
    assert_size_stride(addmm_20, (1568, 2048), (2048, 1))
    assert_size_stride(view_52, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_21, (1568, 512), (512, 1))
    assert_size_stride(add_49, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_14, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_29, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_14, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_55, (1568, 512), (512, 1))
    assert_size_stride(addmm_22, (1568, 2048), (2048, 1))
    assert_size_stride(view_57, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_23, (1568, 512), (512, 1))
    assert_size_stride(add_53, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_15, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_31, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_15, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_60, (1568, 512), (512, 1))
    assert_size_stride(addmm_24, (1568, 2048), (2048, 1))
    assert_size_stride(view_62, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_25, (1568, 512), (512, 1))
    assert_size_stride(add_57, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_16, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_33, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_16, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_65, (1568, 512), (512, 1))
    assert_size_stride(addmm_26, (1568, 2048), (2048, 1))
    assert_size_stride(view_67, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_27, (1568, 512), (512, 1))
    assert_size_stride(add_61, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_17, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_35, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_17, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_70, (1568, 512), (512, 1))
    assert_size_stride(addmm_28, (1568, 2048), (2048, 1))
    assert_size_stride(view_72, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_29, (1568, 512), (512, 1))
    assert_size_stride(add_65, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_18, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_37, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_18, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_75, (1568, 512), (512, 1))
    assert_size_stride(addmm_30, (1568, 2048), (2048, 1))
    assert_size_stride(view_77, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_31, (1568, 512), (512, 1))
    assert_size_stride(add_69, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_19, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_39, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_19, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_80, (1568, 512), (512, 1))
    assert_size_stride(addmm_32, (1568, 2048), (2048, 1))
    assert_size_stride(view_82, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_33, (1568, 512), (512, 1))
    assert_size_stride(add_73, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_20, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_41, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_20, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_85, (1568, 512), (512, 1))
    assert_size_stride(addmm_34, (1568, 2048), (2048, 1))
    assert_size_stride(view_87, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_35, (1568, 512), (512, 1))
    assert_size_stride(add_77, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_21, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_43, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_21, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_90, (1568, 512), (512, 1))
    assert_size_stride(addmm_36, (1568, 2048), (2048, 1))
    assert_size_stride(view_92, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_37, (1568, 512), (512, 1))
    assert_size_stride(add_81, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_22, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_45, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_22, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_95, (1568, 512), (512, 1))
    assert_size_stride(addmm_38, (1568, 2048), (2048, 1))
    assert_size_stride(view_97, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_39, (1568, 512), (512, 1))
    assert_size_stride(add_85, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_23, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_47, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_23, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_100, (1568, 512), (512, 1))
    assert_size_stride(addmm_40, (1568, 2048), (2048, 1))
    assert_size_stride(view_102, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_41, (1568, 512), (512, 1))
    assert_size_stride(add_89, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_24, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_49, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_24, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_105, (1568, 512), (512, 1))
    assert_size_stride(addmm_42, (1568, 2048), (2048, 1))
    assert_size_stride(view_107, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_43, (1568, 512), (512, 1))
    assert_size_stride(add_93, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_25, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_51, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_25, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_110, (1568, 512), (512, 1))
    assert_size_stride(addmm_44, (1568, 2048), (2048, 1))
    assert_size_stride(view_112, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_45, (1568, 512), (512, 1))
    assert_size_stride(add_97, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_26, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_53, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_26, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_115, (1568, 512), (512, 1))
    assert_size_stride(addmm_46, (1568, 2048), (2048, 1))
    assert_size_stride(view_117, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_47, (1568, 512), (512, 1))
    assert_size_stride(add_101, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_27, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_55, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_27, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_120, (1568, 512), (512, 1))
    assert_size_stride(addmm_48, (1568, 2048), (2048, 1))
    assert_size_stride(view_122, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_49, (1568, 512), (512, 1))
    assert_size_stride(add_105, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_28, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_57, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_28, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_125, (1568, 512), (512, 1))
    assert_size_stride(addmm_50, (1568, 2048), (2048, 1))
    assert_size_stride(view_127, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_51, (1568, 512), (512, 1))
    assert_size_stride(add_109, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_29, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_59, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_29, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_130, (1568, 512), (512, 1))
    assert_size_stride(addmm_52, (1568, 2048), (2048, 1))
    assert_size_stride(view_132, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_53, (1568, 512), (512, 1))
    assert_size_stride(add_113, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_30, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_61, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_30, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_135, (1568, 512), (512, 1))
    assert_size_stride(addmm_54, (1568, 2048), (2048, 1))
    assert_size_stride(view_137, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_55, (1568, 512), (512, 1))
    assert_size_stride(add_117, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_31, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_63, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_31, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_140, (1568, 512), (512, 1))
    assert_size_stride(addmm_56, (1568, 2048), (2048, 1))
    assert_size_stride(view_142, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_57, (1568, 512), (512, 1))
    assert_size_stride(add_121, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_32, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_65, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_32, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_145, (1568, 512), (512, 1))
    assert_size_stride(addmm_58, (1568, 2048), (2048, 1))
    assert_size_stride(view_147, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_59, (1568, 512), (512, 1))
    assert_size_stride(add_125, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_33, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_67, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_33, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_150, (1568, 512), (512, 1))
    assert_size_stride(addmm_60, (1568, 2048), (2048, 1))
    assert_size_stride(view_152, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_61, (1568, 512), (512, 1))
    assert_size_stride(add_129, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_34, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_69, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_34, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_155, (1568, 512), (512, 1))
    assert_size_stride(addmm_62, (1568, 2048), (2048, 1))
    assert_size_stride(view_157, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_63, (1568, 512), (512, 1))
    assert_size_stride(add_133, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_35, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(getitem_71, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(rsqrt_35, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(view_160, (1568, 512), (512, 1))
    assert_size_stride(addmm_64, (1568, 2048), (2048, 1))
    assert_size_stride(view_162, (1568, 2048), (2048, 1))
    assert_size_stride(addmm_65, (1568, 512), (512, 1))
    assert_size_stride(mul_204, (8, 14, 14, 512), (100352, 1, 7168, 14))
    assert_size_stride(permute_139, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convolution_36, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(convolution_37, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(getitem_75, (8, 7, 7, 1), (49, 1, 7, 7))
    assert_size_stride(rsqrt_37, (8, 7, 7, 1), (49, 1, 7, 7))
    assert_size_stride(view_165, (392, 1024), (1024, 1))
    assert_size_stride(addmm_66, (392, 4096), (4096, 1))
    assert_size_stride(view_167, (392, 4096), (4096, 1))
    assert_size_stride(addmm_67, (392, 1024), (1024, 1))
    assert_size_stride(add_143, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(convolution_38, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(getitem_77, (8, 7, 7, 1), (49, 1, 7, 7))
    assert_size_stride(rsqrt_38, (8, 7, 7, 1), (49, 1, 7, 7))
    assert_size_stride(view_170, (392, 1024), (1024, 1))
    assert_size_stride(addmm_68, (392, 4096), (4096, 1))
    assert_size_stride(view_172, (392, 4096), (4096, 1))
    assert_size_stride(addmm_69, (392, 1024), (1024, 1))
    assert_size_stride(add_147, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(convolution_39, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    assert_size_stride(getitem_79, (8, 7, 7, 1), (49, 1, 7, 7))
    assert_size_stride(rsqrt_39, (8, 7, 7, 1), (49, 1, 7, 7))
    assert_size_stride(view_175, (392, 1024), (1024, 1))
    assert_size_stride(addmm_70, (392, 4096), (4096, 1))
    assert_size_stride(view_177, (392, 4096), (4096, 1))
    assert_size_stride(addmm_71, (392, 1024), (1024, 1))
    assert_size_stride(mul_224, (8, 1, 1, 1024), (1024, 1, 1024, 1))
    assert_size_stride(clone_73, (8, 1024), (1024, 1))
    assert_size_stride(permute_155, (1000, 1024), (1024, 1))
    assert_size_stride(div, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(permute_162, (1024, 4096), (4096, 1))
    assert_size_stride(permute_166, (4096, 1024), (1024, 1))
    assert_size_stride(permute_172, (1024, 4096), (4096, 1))
    assert_size_stride(permute_176, (4096, 1024), (1024, 1))
    assert_size_stride(permute_182, (1024, 4096), (4096, 1))
    assert_size_stride(permute_186, (4096, 1024), (1024, 1))
    assert_size_stride(div_5, (8, 14, 14, 1), (196, 1, 14, 14))
    assert_size_stride(permute_194, (512, 2048), (2048, 1))
    assert_size_stride(permute_198, (2048, 512), (512, 1))
    assert_size_stride(permute_204, (512, 2048), (2048, 1))
    assert_size_stride(permute_208, (2048, 512), (512, 1))
    assert_size_stride(permute_214, (512, 2048), (2048, 1))
    assert_size_stride(permute_218, (2048, 512), (512, 1))
    assert_size_stride(permute_224, (512, 2048), (2048, 1))
    assert_size_stride(permute_228, (2048, 512), (512, 1))
    assert_size_stride(permute_234, (512, 2048), (2048, 1))
    assert_size_stride(permute_238, (2048, 512), (512, 1))
    assert_size_stride(permute_244, (512, 2048), (2048, 1))
    assert_size_stride(permute_248, (2048, 512), (512, 1))
    assert_size_stride(permute_254, (512, 2048), (2048, 1))
    assert_size_stride(permute_258, (2048, 512), (512, 1))
    assert_size_stride(permute_264, (512, 2048), (2048, 1))
    assert_size_stride(permute_268, (2048, 512), (512, 1))
    assert_size_stride(permute_274, (512, 2048), (2048, 1))
    assert_size_stride(permute_278, (2048, 512), (512, 1))
    assert_size_stride(permute_284, (512, 2048), (2048, 1))
    assert_size_stride(permute_288, (2048, 512), (512, 1))
    assert_size_stride(permute_294, (512, 2048), (2048, 1))
    assert_size_stride(permute_298, (2048, 512), (512, 1))
    assert_size_stride(permute_304, (512, 2048), (2048, 1))
    assert_size_stride(permute_308, (2048, 512), (512, 1))
    assert_size_stride(permute_314, (512, 2048), (2048, 1))
    assert_size_stride(permute_318, (2048, 512), (512, 1))
    assert_size_stride(permute_324, (512, 2048), (2048, 1))
    assert_size_stride(permute_328, (2048, 512), (512, 1))
    assert_size_stride(permute_334, (512, 2048), (2048, 1))
    assert_size_stride(permute_338, (2048, 512), (512, 1))
    assert_size_stride(permute_344, (512, 2048), (2048, 1))
    assert_size_stride(permute_348, (2048, 512), (512, 1))
    assert_size_stride(permute_354, (512, 2048), (2048, 1))
    assert_size_stride(permute_358, (2048, 512), (512, 1))
    assert_size_stride(permute_364, (512, 2048), (2048, 1))
    assert_size_stride(permute_368, (2048, 512), (512, 1))
    assert_size_stride(permute_374, (512, 2048), (2048, 1))
    assert_size_stride(permute_378, (2048, 512), (512, 1))
    assert_size_stride(permute_384, (512, 2048), (2048, 1))
    assert_size_stride(permute_388, (2048, 512), (512, 1))
    assert_size_stride(permute_394, (512, 2048), (2048, 1))
    assert_size_stride(permute_398, (2048, 512), (512, 1))
    assert_size_stride(permute_404, (512, 2048), (2048, 1))
    assert_size_stride(permute_408, (2048, 512), (512, 1))
    assert_size_stride(permute_414, (512, 2048), (2048, 1))
    assert_size_stride(permute_418, (2048, 512), (512, 1))
    assert_size_stride(permute_424, (512, 2048), (2048, 1))
    assert_size_stride(permute_428, (2048, 512), (512, 1))
    assert_size_stride(permute_434, (512, 2048), (2048, 1))
    assert_size_stride(permute_438, (2048, 512), (512, 1))
    assert_size_stride(permute_444, (512, 2048), (2048, 1))
    assert_size_stride(permute_448, (2048, 512), (512, 1))
    assert_size_stride(permute_454, (512, 2048), (2048, 1))
    assert_size_stride(permute_458, (2048, 512), (512, 1))
    assert_size_stride(div_33, (8, 28, 28, 1), (784, 1, 28, 28))
    assert_size_stride(permute_466, (256, 1024), (1024, 1))
    assert_size_stride(permute_470, (1024, 256), (256, 1))
    assert_size_stride(permute_476, (256, 1024), (1024, 1))
    assert_size_stride(permute_480, (1024, 256), (256, 1))
    assert_size_stride(permute_486, (256, 1024), (1024, 1))
    assert_size_stride(permute_490, (1024, 256), (256, 1))
    assert_size_stride(div_37, (8, 56, 56, 1), (3136, 1, 56, 56))
    assert_size_stride(permute_498, (128, 512), (512, 1))
    assert_size_stride(permute_502, (512, 128), (128, 1))
    assert_size_stride(permute_508, (128, 512), (512, 1))
    assert_size_stride(permute_512, (512, 128), (128, 1))
    assert_size_stride(permute_518, (128, 512), (512, 1))
    assert_size_stride(permute_522, (512, 128), (128, 1))
    assert_size_stride(div_41, (8, 56, 56, 1), (3136, 1, 56, 56))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_155, out=buf0)
    del permute_155
    buf1 = empty((1000, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_73, out=buf1)
    del clone_73
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cpu', dtype=torch.float32)
    buf5 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf6 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf7 = buf0; del buf0  # reuse
    buf9 = empty((8192, ), device='cpu', dtype=torch.float32)
    buf12 = empty((8, 7, 7, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_as_strided_scatter_clone_native_layer_norm_backward_squeeze_sum_0(c_void_p(buf7.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(mul_224.data_ptr()), c_void_p(div.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf12.data_ptr()))
    del buf3
    del buf4
    del buf7
    del div
    del mul_224
    del primals_116
    del primals_117
    del tangents_1
    buf13 = empty((392, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf12, (392, 1024), (1024, 1), 0), permute_162, out=buf13)
    del permute_162
    buf16 = reinterpret_tensor(buf13, (8, 7, 7, 4096), (200704, 28672, 4096, 1), 0); del buf13  # reuse
    cpp_fused_gelu_gelu_backward_1(c_void_p(buf16.data_ptr()), c_void_p(addmm_70.data_ptr()))
    del addmm_70
    buf17 = empty((392, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf16, (392, 4096), (4096, 1), 0), permute_166, out=buf17)
    del permute_166
    buf20 = empty_strided((8, 7, 7, 1), (49, 7, 1, 392), device='cpu', dtype=torch.float32)
    buf21 = empty_strided((8, 7, 7, 1), (49, 7, 1, 392), device='cpu', dtype=torch.float32)
    buf24 = empty_strided((8, 1024, 7, 7), (50176, 1, 1024, 7168), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_2(c_void_p(buf17.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(getitem_79.data_ptr()), c_void_p(rsqrt_39.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf24.data_ptr()))
    del primals_114
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf25 = aten.convolution_backward(buf24, add_147, primals_337, [1024], [1, 1], [3, 3], [1, 1], False, [0, 0], 1024, [True, True, True])
    del add_147
    del primals_337
    buf26 = buf25[0]
    buf30 = reinterpret_tensor(buf24, (8, 7, 7, 1024), (50176, 7168, 1024, 1), 0); del buf24  # reuse
    cpp_fused_clone_3(c_void_p(buf9.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf30.data_ptr()))
    del primals_113
    buf31 = empty((392, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf30, (392, 1024), (1024, 1), 0), permute_172, out=buf31)
    del permute_172
    buf34 = reinterpret_tensor(buf31, (8, 7, 7, 4096), (200704, 28672, 4096, 1), 0); del buf31  # reuse
    cpp_fused_gelu_gelu_backward_4(c_void_p(buf34.data_ptr()), c_void_p(addmm_68.data_ptr()))
    del addmm_68
    buf35 = empty((392, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf34, (392, 4096), (4096, 1), 0), permute_176, out=buf35)
    del permute_176
    buf38 = buf21; del buf21  # reuse
    buf39 = buf20; del buf20  # reuse
    buf42 = empty_strided((8, 1024, 7, 7), (50176, 1, 1024, 7168), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_5(c_void_p(buf35.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(getitem_77.data_ptr()), c_void_p(rsqrt_38.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf42.data_ptr()))
    del primals_111
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf43 = aten.convolution_backward(buf42, add_143, primals_331, [1024], [1, 1], [3, 3], [1, 1], False, [0, 0], 1024, [True, True, True])
    del add_143
    del buf42
    del primals_331
    buf44 = buf43[0]
    buf11 = empty((1, 1024, 1, 1), device='cpu', dtype=torch.float32)
    buf29 = empty((1, 1024, 1, 1), device='cpu', dtype=torch.float32)
    buf47 = empty((1, 1024, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_sum_6(c_void_p(buf9.data_ptr()), c_void_p(addmm_71.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(addmm_69.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(addmm_67.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf47.data_ptr()))
    del addmm_67
    del addmm_69
    del addmm_71
    buf14 = empty((1024, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf12, (1024, 392), (1, 1024), 0), view_177, out=buf14)
    del view_177
    buf15 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_7(c_void_p(buf12.data_ptr()), c_void_p(buf15.data_ptr()))
    del buf12
    buf18 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf16, (4096, 392), (1, 4096), 0), view_175, out=buf18)
    del view_175
    buf19 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf22 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf23 = empty((1024, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_8(c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(getitem_79.data_ptr()), c_void_p(rsqrt_39.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()))
    del buf17
    del convolution_39
    del getitem_79
    del rsqrt_39
    buf27 = buf25[1]
    buf28 = buf25[2]
    del buf25
    buf32 = empty((1024, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf30, (1024, 392), (1, 1024), 0), view_172, out=buf32)
    del view_172
    buf33 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_9(c_void_p(buf30.data_ptr()), c_void_p(buf33.data_ptr()))
    buf36 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf34, (4096, 392), (1, 4096), 0), view_170, out=buf36)
    del view_170
    buf37 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf40 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf41 = empty((1024, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_10(c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(getitem_77.data_ptr()), c_void_p(rsqrt_38.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()))
    del convolution_38
    del getitem_77
    del rsqrt_38
    buf45 = buf43[1]
    buf46 = buf43[2]
    del buf43
    buf48 = reinterpret_tensor(buf35, (8, 7, 7, 1024), (50176, 7168, 1024, 1), 0); del buf35  # reuse
    cpp_fused_clone_11(c_void_p(buf9.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf48.data_ptr()))
    del primals_110
    buf49 = reinterpret_tensor(buf34, (392, 4096), (4096, 1), 0); del buf34  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf48, (392, 1024), (1024, 1), 0), permute_182, out=buf49)
    del permute_182
    buf50 = empty((1024, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf48, (1024, 392), (1, 1024), 0), view_167, out=buf50)
    del view_167
    buf51 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf52 = reinterpret_tensor(buf49, (8, 7, 7, 4096), (200704, 28672, 4096, 1), 0); del buf49  # reuse
    cpp_fused_gelu_gelu_backward_sum_12(c_void_p(buf52.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(addmm_66.data_ptr()), c_void_p(buf51.data_ptr()))
    del addmm_66
    buf53 = reinterpret_tensor(buf48, (392, 1024), (1024, 1), 0); del buf48  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf52, (392, 4096), (4096, 1), 0), permute_186, out=buf53)
    del permute_186
    buf54 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf52, (4096, 392), (1, 4096), 0), view_165, out=buf54)
    del view_165
    buf55 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf56 = buf39; del buf39  # reuse
    buf57 = buf38; del buf38  # reuse
    buf58 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf59 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf60 = reinterpret_tensor(buf30, (8, 1024, 7, 7), (50176, 1, 1024, 7168), 0); del buf30  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_sum_13(c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(getitem_75.data_ptr()), c_void_p(rsqrt_37.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()))
    del buf53
    del buf56
    del buf57
    del convolution_37
    del getitem_75
    del primals_108
    del rsqrt_37
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf61 = aten.convolution_backward(buf60, convolution_36, primals_325, [1024], [1, 1], [3, 3], [1, 1], False, [0, 0], 1024, [True, True, True])
    del convolution_36
    del primals_325
    buf62 = buf61[0]
    buf63 = buf61[1]
    buf64 = buf61[2]
    del buf61
    buf65 = reinterpret_tensor(buf60, (8, 1024, 7, 7), (50176, 49, 7, 1), 0); del buf60  # reuse
    cpp_fused_add_convolution_backward_div_14(c_void_p(buf9.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf65.data_ptr()))
    del buf26
    del buf44
    del buf62
    del buf9
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div]
    buf66 = aten.convolution_backward(buf65, permute_139, primals_323, [1024], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf65
    del permute_139
    del primals_323
    buf67 = buf66[0]
    buf68 = buf66[1]
    buf69 = buf66[2]
    del buf66
    buf70 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf71 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf72 = empty((512, ), device='cpu', dtype=torch.float32)
    buf73 = empty((512, ), device='cpu', dtype=torch.float32)
    buf74 = reinterpret_tensor(buf67, (8, 14, 14, 512), (100352, 7168, 512, 1), 0); del buf67  # reuse
    buf76 = empty((8, 512, 14, 14), device='cpu', dtype=torch.float32)
    buf77 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_native_layer_norm_backward_view_15(c_void_p(buf74.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(mul_204.data_ptr()), c_void_p(div_5.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()))
    del div_5
    del mul_204
    del primals_105
    del primals_106
    buf78 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf77, permute_194, out=buf78)
    del permute_194
    buf81 = reinterpret_tensor(buf78, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf78  # reuse
    cpp_fused_gelu_gelu_backward_16(c_void_p(buf81.data_ptr()), c_void_p(addmm_64.data_ptr()))
    del addmm_64
    buf82 = reinterpret_tensor(buf76, (1568, 512), (512, 1), 0); del buf76  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf81, (1568, 2048), (2048, 1), 0), permute_198, out=buf82)
    del permute_198
    buf85 = buf71; del buf71  # reuse
    buf86 = buf70; del buf70  # reuse
    buf89 = empty_strided((8, 512, 14, 14), (100352, 1, 512, 7168), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_17(c_void_p(buf82.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(getitem_71.data_ptr()), c_void_p(rsqrt_35.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf89.data_ptr()))
    del primals_103
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf90 = aten.convolution_backward(buf89, add_133, primals_317, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_133
    del primals_317
    buf91 = buf90[0]
    buf95 = reinterpret_tensor(buf89, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf89  # reuse
    buf96 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_view_18(c_void_p(buf74.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()))
    del primals_102
    buf97 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf96, permute_204, out=buf97)
    del permute_204
    buf100 = reinterpret_tensor(buf97, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf97  # reuse
    cpp_fused_gelu_gelu_backward_19(c_void_p(buf100.data_ptr()), c_void_p(addmm_62.data_ptr()))
    del addmm_62
    buf101 = reinterpret_tensor(buf95, (1568, 512), (512, 1), 0); del buf95  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf100, (1568, 2048), (2048, 1), 0), permute_208, out=buf101)
    del permute_208
    buf104 = buf86; del buf86  # reuse
    buf105 = buf85; del buf85  # reuse
    buf108 = empty_strided((8, 512, 14, 14), (100352, 1, 512, 7168), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_20(c_void_p(buf101.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(getitem_69.data_ptr()), c_void_p(rsqrt_34.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf108.data_ptr()))
    del primals_100
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf109 = aten.convolution_backward(buf108, add_129, primals_311, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_129
    del primals_311
    buf110 = buf109[0]
    buf114 = reinterpret_tensor(buf108, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf108  # reuse
    buf115 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_view_21(c_void_p(buf74.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()))
    del primals_99
    buf116 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf115, permute_214, out=buf116)
    del permute_214
    buf119 = reinterpret_tensor(buf116, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf116  # reuse
    cpp_fused_gelu_gelu_backward_22(c_void_p(buf119.data_ptr()), c_void_p(addmm_60.data_ptr()))
    del addmm_60
    buf120 = reinterpret_tensor(buf114, (1568, 512), (512, 1), 0); del buf114  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf119, (1568, 2048), (2048, 1), 0), permute_218, out=buf120)
    del permute_218
    buf123 = buf105; del buf105  # reuse
    buf124 = buf104; del buf104  # reuse
    buf127 = empty_strided((8, 512, 14, 14), (100352, 1, 512, 7168), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_23(c_void_p(buf120.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(getitem_67.data_ptr()), c_void_p(rsqrt_33.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf127.data_ptr()))
    del primals_97
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf128 = aten.convolution_backward(buf127, add_125, primals_305, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_125
    del buf127
    del primals_305
    buf129 = buf128[0]
    buf75 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf94 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf113 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf132 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_24(c_void_p(buf74.data_ptr()), c_void_p(addmm_65.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(addmm_63.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(addmm_61.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(addmm_59.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf132.data_ptr()))
    del addmm_59
    del addmm_61
    del addmm_63
    del addmm_65
    buf79 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf77, (512, 1568), (1, 512), 0), view_162, out=buf79)
    del view_162
    buf80 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_25(c_void_p(buf77.data_ptr()), c_void_p(buf80.data_ptr()))
    del buf77
    buf83 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf81, (2048, 1568), (1, 2048), 0), view_160, out=buf83)
    del view_160
    buf84 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf87 = empty((512, ), device='cpu', dtype=torch.float32)
    buf88 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_26(c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(getitem_71.data_ptr()), c_void_p(rsqrt_35.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()))
    del buf82
    del convolution_35
    del getitem_71
    del rsqrt_35
    buf92 = buf90[1]
    buf93 = buf90[2]
    del buf90
    buf98 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf96, (512, 1568), (1, 512), 0), view_157, out=buf98)
    del view_157
    buf99 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_27(c_void_p(buf96.data_ptr()), c_void_p(buf99.data_ptr()))
    del buf96
    buf102 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf100, (2048, 1568), (1, 2048), 0), view_155, out=buf102)
    del view_155
    buf103 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf106 = empty((512, ), device='cpu', dtype=torch.float32)
    buf107 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_28(c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(getitem_69.data_ptr()), c_void_p(rsqrt_34.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()))
    del convolution_34
    del getitem_69
    del rsqrt_34
    buf111 = buf109[1]
    buf112 = buf109[2]
    del buf109
    buf117 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf115, (512, 1568), (1, 512), 0), view_152, out=buf117)
    del view_152
    buf118 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_29(c_void_p(buf115.data_ptr()), c_void_p(buf118.data_ptr()))
    buf121 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf119, (2048, 1568), (1, 2048), 0), view_150, out=buf121)
    del view_150
    buf122 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf125 = empty((512, ), device='cpu', dtype=torch.float32)
    buf126 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_30(c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(getitem_67.data_ptr()), c_void_p(rsqrt_33.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()))
    del convolution_33
    del getitem_67
    del rsqrt_33
    buf130 = buf128[1]
    buf131 = buf128[2]
    del buf128
    buf133 = reinterpret_tensor(buf120, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf120  # reuse
    buf134 = buf115; del buf115  # reuse
    cpp_fused_add_mul_view_31(c_void_p(buf74.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()))
    del primals_96
    buf135 = reinterpret_tensor(buf119, (1568, 2048), (2048, 1), 0); del buf119  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf134, permute_224, out=buf135)
    del permute_224
    buf136 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf134, (512, 1568), (1, 512), 0), view_147, out=buf136)
    del view_147
    buf137 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf138 = reinterpret_tensor(buf135, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf135  # reuse
    cpp_fused_gelu_gelu_backward_sum_32(c_void_p(buf138.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(addmm_58.data_ptr()), c_void_p(buf137.data_ptr()))
    del addmm_58
    buf139 = buf134; del buf134  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf138, (1568, 2048), (2048, 1), 0), permute_228, out=buf139)
    del permute_228
    buf140 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf138, (2048, 1568), (1, 2048), 0), view_145, out=buf140)
    del view_145
    buf141 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf142 = buf124; del buf124  # reuse
    buf143 = buf123; del buf123  # reuse
    buf144 = empty((512, ), device='cpu', dtype=torch.float32)
    buf145 = empty((512, ), device='cpu', dtype=torch.float32)
    buf146 = reinterpret_tensor(buf133, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf133  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_sum_33(c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(getitem_65.data_ptr()), c_void_p(rsqrt_32.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()))
    del convolution_32
    del getitem_65
    del primals_94
    del rsqrt_32
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf147 = aten.convolution_backward(buf146, add_121, primals_299, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_121
    del primals_299
    buf148 = buf147[0]
    buf149 = buf147[1]
    buf150 = buf147[2]
    del buf147
    buf151 = buf110; del buf110  # reuse
    buf153 = reinterpret_tensor(buf146, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf146  # reuse
    buf154 = buf139; del buf139  # reuse
    cpp_fused_add_mul_view_34(c_void_p(buf151.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()))
    del primals_93
    buf155 = reinterpret_tensor(buf138, (1568, 2048), (2048, 1), 0); del buf138  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf154, permute_234, out=buf155)
    del permute_234
    buf158 = reinterpret_tensor(buf155, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf155  # reuse
    cpp_fused_gelu_gelu_backward_35(c_void_p(buf158.data_ptr()), c_void_p(addmm_56.data_ptr()))
    del addmm_56
    buf159 = reinterpret_tensor(buf91, (1568, 512), (512, 1), 0); del buf91  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf158, (1568, 2048), (2048, 1), 0), permute_238, out=buf159)
    del permute_238
    buf162 = buf143; del buf143  # reuse
    buf163 = buf142; del buf142  # reuse
    buf166 = reinterpret_tensor(buf74, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf74  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_36(c_void_p(buf159.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(getitem_63.data_ptr()), c_void_p(rsqrt_31.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf166.data_ptr()))
    del primals_91
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf167 = aten.convolution_backward(buf166, add_117, primals_293, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_117
    del primals_293
    buf168 = buf167[0]
    buf172 = reinterpret_tensor(buf166, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf166  # reuse
    buf173 = reinterpret_tensor(buf153, (1568, 512), (512, 1), 0); del buf153  # reuse
    cpp_fused_add_mul_view_37(c_void_p(buf151.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()))
    del primals_90
    buf174 = reinterpret_tensor(buf100, (1568, 2048), (2048, 1), 0); del buf100  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf173, permute_244, out=buf174)
    del permute_244
    buf177 = reinterpret_tensor(buf174, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf174  # reuse
    cpp_fused_gelu_gelu_backward_38(c_void_p(buf177.data_ptr()), c_void_p(addmm_54.data_ptr()))
    del addmm_54
    buf178 = reinterpret_tensor(buf172, (1568, 512), (512, 1), 0); del buf172  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf177, (1568, 2048), (2048, 1), 0), permute_248, out=buf178)
    del permute_248
    buf181 = buf163; del buf163  # reuse
    buf182 = buf162; del buf162  # reuse
    buf185 = reinterpret_tensor(buf148, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf148  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_39(c_void_p(buf178.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(getitem_61.data_ptr()), c_void_p(rsqrt_30.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf185.data_ptr()))
    del primals_88
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf186 = aten.convolution_backward(buf185, add_113, primals_287, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_113
    del primals_287
    buf187 = buf186[0]
    buf191 = reinterpret_tensor(buf185, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf185  # reuse
    buf192 = reinterpret_tensor(buf129, (1568, 512), (512, 1), 0); del buf129  # reuse
    cpp_fused_add_mul_view_40(c_void_p(buf151.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()))
    del primals_87
    buf193 = reinterpret_tensor(buf81, (1568, 2048), (2048, 1), 0); del buf81  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf192, permute_254, out=buf193)
    del permute_254
    buf196 = reinterpret_tensor(buf193, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf193  # reuse
    cpp_fused_gelu_gelu_backward_41(c_void_p(buf196.data_ptr()), c_void_p(addmm_52.data_ptr()))
    del addmm_52
    buf197 = reinterpret_tensor(buf191, (1568, 512), (512, 1), 0); del buf191  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf196, (1568, 2048), (2048, 1), 0), permute_258, out=buf197)
    del permute_258
    buf200 = buf182; del buf182  # reuse
    buf201 = buf181; del buf181  # reuse
    buf204 = reinterpret_tensor(buf101, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf101  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_42(c_void_p(buf197.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(getitem_59.data_ptr()), c_void_p(rsqrt_29.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf204.data_ptr()))
    del primals_85
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf205 = aten.convolution_backward(buf204, add_109, primals_281, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_109
    del buf204
    del primals_281
    buf206 = buf205[0]
    buf152 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf171 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf190 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf209 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_43(c_void_p(buf151.data_ptr()), c_void_p(addmm_57.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(addmm_55.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(addmm_53.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(addmm_51.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf209.data_ptr()))
    del addmm_51
    del addmm_53
    del addmm_55
    del addmm_57
    buf156 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf154, (512, 1568), (1, 512), 0), view_142, out=buf156)
    del view_142
    buf157 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_44(c_void_p(buf154.data_ptr()), c_void_p(buf157.data_ptr()))
    del buf154
    buf160 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf158, (2048, 1568), (1, 2048), 0), view_140, out=buf160)
    del view_140
    buf161 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf164 = empty((512, ), device='cpu', dtype=torch.float32)
    buf165 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_45(c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(getitem_63.data_ptr()), c_void_p(rsqrt_31.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()))
    del buf159
    del convolution_31
    del getitem_63
    del rsqrt_31
    buf169 = buf167[1]
    buf170 = buf167[2]
    del buf167
    buf175 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf173, (512, 1568), (1, 512), 0), view_137, out=buf175)
    del view_137
    buf176 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_46(c_void_p(buf173.data_ptr()), c_void_p(buf176.data_ptr()))
    del buf173
    buf179 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf177, (2048, 1568), (1, 2048), 0), view_135, out=buf179)
    del view_135
    buf180 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf183 = empty((512, ), device='cpu', dtype=torch.float32)
    buf184 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_47(c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(getitem_61.data_ptr()), c_void_p(rsqrt_30.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()))
    del convolution_30
    del getitem_61
    del rsqrt_30
    buf188 = buf186[1]
    buf189 = buf186[2]
    del buf186
    buf194 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf192, (512, 1568), (1, 512), 0), view_132, out=buf194)
    del view_132
    buf195 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_48(c_void_p(buf192.data_ptr()), c_void_p(buf195.data_ptr()))
    buf198 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf196, (2048, 1568), (1, 2048), 0), view_130, out=buf198)
    del view_130
    buf199 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf202 = empty((512, ), device='cpu', dtype=torch.float32)
    buf203 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_49(c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(getitem_59.data_ptr()), c_void_p(rsqrt_29.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()))
    del convolution_29
    del getitem_59
    del rsqrt_29
    buf207 = buf205[1]
    buf208 = buf205[2]
    del buf205
    buf210 = reinterpret_tensor(buf197, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf197  # reuse
    buf211 = buf192; del buf192  # reuse
    cpp_fused_add_mul_view_50(c_void_p(buf151.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()))
    del primals_84
    buf212 = reinterpret_tensor(buf196, (1568, 2048), (2048, 1), 0); del buf196  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf211, permute_264, out=buf212)
    del permute_264
    buf213 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf211, (512, 1568), (1, 512), 0), view_127, out=buf213)
    del view_127
    buf214 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf215 = reinterpret_tensor(buf212, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf212  # reuse
    cpp_fused_gelu_gelu_backward_sum_51(c_void_p(buf215.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(addmm_50.data_ptr()), c_void_p(buf214.data_ptr()))
    del addmm_50
    buf216 = buf211; del buf211  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf215, (1568, 2048), (2048, 1), 0), permute_268, out=buf216)
    del permute_268
    buf217 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf215, (2048, 1568), (1, 2048), 0), view_125, out=buf217)
    del view_125
    buf218 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf219 = buf201; del buf201  # reuse
    buf220 = buf200; del buf200  # reuse
    buf221 = empty((512, ), device='cpu', dtype=torch.float32)
    buf222 = empty((512, ), device='cpu', dtype=torch.float32)
    buf223 = reinterpret_tensor(buf210, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf210  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_sum_52(c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(getitem_57.data_ptr()), c_void_p(rsqrt_28.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()))
    del convolution_28
    del getitem_57
    del primals_82
    del rsqrt_28
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf224 = aten.convolution_backward(buf223, add_105, primals_275, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_105
    del primals_275
    buf225 = buf224[0]
    buf226 = buf224[1]
    buf227 = buf224[2]
    del buf224
    buf228 = buf151; del buf151  # reuse
    buf230 = reinterpret_tensor(buf223, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf223  # reuse
    buf231 = buf216; del buf216  # reuse
    cpp_fused_add_mul_view_53(c_void_p(buf228.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()))
    del primals_81
    buf232 = reinterpret_tensor(buf215, (1568, 2048), (2048, 1), 0); del buf215  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf231, permute_274, out=buf232)
    del permute_274
    buf235 = reinterpret_tensor(buf232, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf232  # reuse
    cpp_fused_gelu_gelu_backward_54(c_void_p(buf235.data_ptr()), c_void_p(addmm_48.data_ptr()))
    del addmm_48
    buf236 = reinterpret_tensor(buf230, (1568, 512), (512, 1), 0); del buf230  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf235, (1568, 2048), (2048, 1), 0), permute_278, out=buf236)
    del permute_278
    buf239 = buf220; del buf220  # reuse
    buf240 = buf219; del buf219  # reuse
    buf243 = reinterpret_tensor(buf225, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf225  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_55(c_void_p(buf236.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(getitem_55.data_ptr()), c_void_p(rsqrt_27.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf243.data_ptr()))
    del primals_79
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf244 = aten.convolution_backward(buf243, add_101, primals_269, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_101
    del primals_269
    buf245 = buf244[0]
    buf249 = reinterpret_tensor(buf243, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf243  # reuse
    buf250 = reinterpret_tensor(buf206, (1568, 512), (512, 1), 0); del buf206  # reuse
    cpp_fused_add_mul_view_56(c_void_p(buf228.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()))
    del primals_78
    buf251 = reinterpret_tensor(buf177, (1568, 2048), (2048, 1), 0); del buf177  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf250, permute_284, out=buf251)
    del permute_284
    buf254 = reinterpret_tensor(buf251, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf251  # reuse
    cpp_fused_gelu_gelu_backward_57(c_void_p(buf254.data_ptr()), c_void_p(addmm_46.data_ptr()))
    del addmm_46
    buf255 = reinterpret_tensor(buf249, (1568, 512), (512, 1), 0); del buf249  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf254, (1568, 2048), (2048, 1), 0), permute_288, out=buf255)
    del permute_288
    buf258 = buf240; del buf240  # reuse
    buf259 = buf239; del buf239  # reuse
    buf262 = reinterpret_tensor(buf187, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf187  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_58(c_void_p(buf255.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(getitem_53.data_ptr()), c_void_p(rsqrt_26.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf262.data_ptr()))
    del primals_76
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf263 = aten.convolution_backward(buf262, add_97, primals_263, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_97
    del primals_263
    buf264 = buf263[0]
    buf268 = reinterpret_tensor(buf262, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf262  # reuse
    buf269 = reinterpret_tensor(buf168, (1568, 512), (512, 1), 0); del buf168  # reuse
    cpp_fused_add_mul_view_59(c_void_p(buf228.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()))
    del primals_75
    buf270 = reinterpret_tensor(buf158, (1568, 2048), (2048, 1), 0); del buf158  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf269, permute_294, out=buf270)
    del permute_294
    buf273 = reinterpret_tensor(buf270, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf270  # reuse
    cpp_fused_gelu_gelu_backward_60(c_void_p(buf273.data_ptr()), c_void_p(addmm_44.data_ptr()))
    del addmm_44
    buf274 = reinterpret_tensor(buf268, (1568, 512), (512, 1), 0); del buf268  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf273, (1568, 2048), (2048, 1), 0), permute_298, out=buf274)
    del permute_298
    buf277 = buf259; del buf259  # reuse
    buf278 = buf258; del buf258  # reuse
    buf281 = reinterpret_tensor(buf178, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf178  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_61(c_void_p(buf274.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(getitem_51.data_ptr()), c_void_p(rsqrt_25.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf281.data_ptr()))
    del primals_73
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf282 = aten.convolution_backward(buf281, add_93, primals_257, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_93
    del buf281
    del primals_257
    buf283 = buf282[0]
    buf229 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf248 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf267 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf286 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_62(c_void_p(buf228.data_ptr()), c_void_p(addmm_49.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(addmm_47.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(addmm_45.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(addmm_43.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf286.data_ptr()))
    del addmm_43
    del addmm_45
    del addmm_47
    del addmm_49
    buf233 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf231, (512, 1568), (1, 512), 0), view_122, out=buf233)
    del view_122
    buf234 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_63(c_void_p(buf231.data_ptr()), c_void_p(buf234.data_ptr()))
    del buf231
    buf237 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf235, (2048, 1568), (1, 2048), 0), view_120, out=buf237)
    del view_120
    buf238 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf241 = empty((512, ), device='cpu', dtype=torch.float32)
    buf242 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_64(c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(getitem_55.data_ptr()), c_void_p(rsqrt_27.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()))
    del buf236
    del convolution_27
    del getitem_55
    del rsqrt_27
    buf246 = buf244[1]
    buf247 = buf244[2]
    del buf244
    buf252 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf250, (512, 1568), (1, 512), 0), view_117, out=buf252)
    del view_117
    buf253 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_65(c_void_p(buf250.data_ptr()), c_void_p(buf253.data_ptr()))
    del buf250
    buf256 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf254, (2048, 1568), (1, 2048), 0), view_115, out=buf256)
    del view_115
    buf257 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf260 = empty((512, ), device='cpu', dtype=torch.float32)
    buf261 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_66(c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(getitem_53.data_ptr()), c_void_p(rsqrt_26.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()))
    del convolution_26
    del getitem_53
    del rsqrt_26
    buf265 = buf263[1]
    buf266 = buf263[2]
    del buf263
    buf271 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf269, (512, 1568), (1, 512), 0), view_112, out=buf271)
    del view_112
    buf272 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_67(c_void_p(buf269.data_ptr()), c_void_p(buf272.data_ptr()))
    buf275 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf273, (2048, 1568), (1, 2048), 0), view_110, out=buf275)
    del view_110
    buf276 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf279 = empty((512, ), device='cpu', dtype=torch.float32)
    buf280 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_68(c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(getitem_51.data_ptr()), c_void_p(rsqrt_25.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()))
    del convolution_25
    del getitem_51
    del rsqrt_25
    buf284 = buf282[1]
    buf285 = buf282[2]
    del buf282
    buf287 = reinterpret_tensor(buf274, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf274  # reuse
    buf288 = buf269; del buf269  # reuse
    cpp_fused_add_mul_view_69(c_void_p(buf228.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()))
    del primals_72
    buf289 = reinterpret_tensor(buf273, (1568, 2048), (2048, 1), 0); del buf273  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf288, permute_304, out=buf289)
    del permute_304
    buf290 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf288, (512, 1568), (1, 512), 0), view_107, out=buf290)
    del view_107
    buf291 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf292 = reinterpret_tensor(buf289, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf289  # reuse
    cpp_fused_gelu_gelu_backward_sum_70(c_void_p(buf292.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(addmm_42.data_ptr()), c_void_p(buf291.data_ptr()))
    del addmm_42
    buf293 = buf288; del buf288  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf292, (1568, 2048), (2048, 1), 0), permute_308, out=buf293)
    del permute_308
    buf294 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf292, (2048, 1568), (1, 2048), 0), view_105, out=buf294)
    del view_105
    buf295 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf296 = buf278; del buf278  # reuse
    buf297 = buf277; del buf277  # reuse
    buf298 = empty((512, ), device='cpu', dtype=torch.float32)
    buf299 = empty((512, ), device='cpu', dtype=torch.float32)
    buf300 = reinterpret_tensor(buf287, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf287  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_sum_71(c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(getitem_49.data_ptr()), c_void_p(rsqrt_24.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()))
    del convolution_24
    del getitem_49
    del primals_70
    del rsqrt_24
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf301 = aten.convolution_backward(buf300, add_89, primals_251, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_89
    del primals_251
    buf302 = buf301[0]
    buf303 = buf301[1]
    buf304 = buf301[2]
    del buf301
    buf305 = buf228; del buf228  # reuse
    buf307 = reinterpret_tensor(buf300, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf300  # reuse
    buf308 = buf293; del buf293  # reuse
    cpp_fused_add_mul_view_72(c_void_p(buf305.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()))
    del primals_69
    buf309 = reinterpret_tensor(buf292, (1568, 2048), (2048, 1), 0); del buf292  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf308, permute_314, out=buf309)
    del permute_314
    buf312 = reinterpret_tensor(buf309, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf309  # reuse
    cpp_fused_gelu_gelu_backward_73(c_void_p(buf312.data_ptr()), c_void_p(addmm_40.data_ptr()))
    del addmm_40
    buf313 = reinterpret_tensor(buf307, (1568, 512), (512, 1), 0); del buf307  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf312, (1568, 2048), (2048, 1), 0), permute_318, out=buf313)
    del permute_318
    buf316 = buf297; del buf297  # reuse
    buf317 = buf296; del buf296  # reuse
    buf320 = reinterpret_tensor(buf302, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf302  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_74(c_void_p(buf313.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(getitem_47.data_ptr()), c_void_p(rsqrt_23.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf320.data_ptr()))
    del primals_67
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf321 = aten.convolution_backward(buf320, add_85, primals_245, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_85
    del primals_245
    buf322 = buf321[0]
    buf326 = reinterpret_tensor(buf320, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf320  # reuse
    buf327 = reinterpret_tensor(buf283, (1568, 512), (512, 1), 0); del buf283  # reuse
    cpp_fused_add_mul_view_75(c_void_p(buf305.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()))
    del primals_66
    buf328 = reinterpret_tensor(buf254, (1568, 2048), (2048, 1), 0); del buf254  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf327, permute_324, out=buf328)
    del permute_324
    buf331 = reinterpret_tensor(buf328, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf328  # reuse
    cpp_fused_gelu_gelu_backward_76(c_void_p(buf331.data_ptr()), c_void_p(addmm_38.data_ptr()))
    del addmm_38
    buf332 = reinterpret_tensor(buf326, (1568, 512), (512, 1), 0); del buf326  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf331, (1568, 2048), (2048, 1), 0), permute_328, out=buf332)
    del permute_328
    buf335 = buf317; del buf317  # reuse
    buf336 = buf316; del buf316  # reuse
    buf339 = reinterpret_tensor(buf264, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf264  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_77(c_void_p(buf332.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(getitem_45.data_ptr()), c_void_p(rsqrt_22.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf339.data_ptr()))
    del primals_64
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf340 = aten.convolution_backward(buf339, add_81, primals_239, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_81
    del primals_239
    buf341 = buf340[0]
    buf345 = reinterpret_tensor(buf339, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf339  # reuse
    buf346 = reinterpret_tensor(buf245, (1568, 512), (512, 1), 0); del buf245  # reuse
    cpp_fused_add_mul_view_78(c_void_p(buf305.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()))
    del primals_63
    buf347 = reinterpret_tensor(buf235, (1568, 2048), (2048, 1), 0); del buf235  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf346, permute_334, out=buf347)
    del permute_334
    buf350 = reinterpret_tensor(buf347, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf347  # reuse
    cpp_fused_gelu_gelu_backward_79(c_void_p(buf350.data_ptr()), c_void_p(addmm_36.data_ptr()))
    del addmm_36
    buf351 = reinterpret_tensor(buf345, (1568, 512), (512, 1), 0); del buf345  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf350, (1568, 2048), (2048, 1), 0), permute_338, out=buf351)
    del permute_338
    buf354 = buf336; del buf336  # reuse
    buf355 = buf335; del buf335  # reuse
    buf358 = reinterpret_tensor(buf255, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf255  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_80(c_void_p(buf351.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(getitem_43.data_ptr()), c_void_p(rsqrt_21.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf358.data_ptr()))
    del primals_61
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf359 = aten.convolution_backward(buf358, add_77, primals_233, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_77
    del buf358
    del primals_233
    buf360 = buf359[0]
    buf306 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf325 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf344 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf363 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_81(c_void_p(buf305.data_ptr()), c_void_p(addmm_41.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(addmm_39.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(addmm_37.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(addmm_35.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf363.data_ptr()))
    del addmm_35
    del addmm_37
    del addmm_39
    del addmm_41
    buf310 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf308, (512, 1568), (1, 512), 0), view_102, out=buf310)
    del view_102
    buf311 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_82(c_void_p(buf308.data_ptr()), c_void_p(buf311.data_ptr()))
    del buf308
    buf314 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf312, (2048, 1568), (1, 2048), 0), view_100, out=buf314)
    del view_100
    buf315 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf318 = empty((512, ), device='cpu', dtype=torch.float32)
    buf319 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_83(c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(getitem_47.data_ptr()), c_void_p(rsqrt_23.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()))
    del buf313
    del convolution_23
    del getitem_47
    del rsqrt_23
    buf323 = buf321[1]
    buf324 = buf321[2]
    del buf321
    buf329 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf327, (512, 1568), (1, 512), 0), view_97, out=buf329)
    del view_97
    buf330 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_84(c_void_p(buf327.data_ptr()), c_void_p(buf330.data_ptr()))
    del buf327
    buf333 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf331, (2048, 1568), (1, 2048), 0), view_95, out=buf333)
    del view_95
    buf334 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf337 = empty((512, ), device='cpu', dtype=torch.float32)
    buf338 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_85(c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(getitem_45.data_ptr()), c_void_p(rsqrt_22.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf338.data_ptr()))
    del convolution_22
    del getitem_45
    del rsqrt_22
    buf342 = buf340[1]
    buf343 = buf340[2]
    del buf340
    buf348 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf346, (512, 1568), (1, 512), 0), view_92, out=buf348)
    del view_92
    buf349 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_86(c_void_p(buf346.data_ptr()), c_void_p(buf349.data_ptr()))
    buf352 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf350, (2048, 1568), (1, 2048), 0), view_90, out=buf352)
    del view_90
    buf353 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf356 = empty((512, ), device='cpu', dtype=torch.float32)
    buf357 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_87(c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(getitem_43.data_ptr()), c_void_p(rsqrt_21.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()))
    del convolution_21
    del getitem_43
    del rsqrt_21
    buf361 = buf359[1]
    buf362 = buf359[2]
    del buf359
    buf364 = reinterpret_tensor(buf351, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf351  # reuse
    buf365 = buf346; del buf346  # reuse
    cpp_fused_add_mul_view_88(c_void_p(buf305.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()))
    del primals_60
    buf366 = reinterpret_tensor(buf350, (1568, 2048), (2048, 1), 0); del buf350  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf365, permute_344, out=buf366)
    del permute_344
    buf367 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf365, (512, 1568), (1, 512), 0), view_87, out=buf367)
    del view_87
    buf368 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf369 = reinterpret_tensor(buf366, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf366  # reuse
    cpp_fused_gelu_gelu_backward_sum_89(c_void_p(buf369.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(buf368.data_ptr()))
    del addmm_34
    buf370 = buf365; del buf365  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf369, (1568, 2048), (2048, 1), 0), permute_348, out=buf370)
    del permute_348
    buf371 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf369, (2048, 1568), (1, 2048), 0), view_85, out=buf371)
    del view_85
    buf372 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf373 = buf355; del buf355  # reuse
    buf374 = buf354; del buf354  # reuse
    buf375 = empty((512, ), device='cpu', dtype=torch.float32)
    buf376 = empty((512, ), device='cpu', dtype=torch.float32)
    buf377 = reinterpret_tensor(buf364, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf364  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_sum_90(c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(getitem_41.data_ptr()), c_void_p(rsqrt_20.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()))
    del convolution_20
    del getitem_41
    del primals_58
    del rsqrt_20
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf378 = aten.convolution_backward(buf377, add_73, primals_227, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_73
    del primals_227
    buf379 = buf378[0]
    buf380 = buf378[1]
    buf381 = buf378[2]
    del buf378
    buf382 = buf305; del buf305  # reuse
    buf384 = reinterpret_tensor(buf377, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf377  # reuse
    buf385 = buf370; del buf370  # reuse
    cpp_fused_add_mul_view_91(c_void_p(buf382.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()))
    del primals_57
    buf386 = reinterpret_tensor(buf369, (1568, 2048), (2048, 1), 0); del buf369  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf385, permute_354, out=buf386)
    del permute_354
    buf389 = reinterpret_tensor(buf386, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf386  # reuse
    cpp_fused_gelu_gelu_backward_92(c_void_p(buf389.data_ptr()), c_void_p(addmm_32.data_ptr()))
    del addmm_32
    buf390 = reinterpret_tensor(buf384, (1568, 512), (512, 1), 0); del buf384  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf389, (1568, 2048), (2048, 1), 0), permute_358, out=buf390)
    del permute_358
    buf393 = buf374; del buf374  # reuse
    buf394 = buf373; del buf373  # reuse
    buf397 = reinterpret_tensor(buf379, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf379  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_93(c_void_p(buf390.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(getitem_39.data_ptr()), c_void_p(rsqrt_19.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf397.data_ptr()))
    del primals_55
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf398 = aten.convolution_backward(buf397, add_69, primals_221, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_69
    del primals_221
    buf399 = buf398[0]
    buf403 = reinterpret_tensor(buf397, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf397  # reuse
    buf404 = reinterpret_tensor(buf360, (1568, 512), (512, 1), 0); del buf360  # reuse
    cpp_fused_add_mul_view_94(c_void_p(buf382.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()))
    del primals_54
    buf405 = reinterpret_tensor(buf331, (1568, 2048), (2048, 1), 0); del buf331  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf404, permute_364, out=buf405)
    del permute_364
    buf408 = reinterpret_tensor(buf405, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf405  # reuse
    cpp_fused_gelu_gelu_backward_95(c_void_p(buf408.data_ptr()), c_void_p(addmm_30.data_ptr()))
    del addmm_30
    buf409 = reinterpret_tensor(buf403, (1568, 512), (512, 1), 0); del buf403  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf408, (1568, 2048), (2048, 1), 0), permute_368, out=buf409)
    del permute_368
    buf412 = buf394; del buf394  # reuse
    buf413 = buf393; del buf393  # reuse
    buf416 = reinterpret_tensor(buf341, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf341  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_96(c_void_p(buf409.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(getitem_37.data_ptr()), c_void_p(rsqrt_18.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf416.data_ptr()))
    del primals_52
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf417 = aten.convolution_backward(buf416, add_65, primals_215, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_65
    del primals_215
    buf418 = buf417[0]
    buf422 = reinterpret_tensor(buf416, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf416  # reuse
    buf423 = reinterpret_tensor(buf322, (1568, 512), (512, 1), 0); del buf322  # reuse
    cpp_fused_add_mul_view_97(c_void_p(buf382.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf423.data_ptr()))
    del primals_51
    buf424 = reinterpret_tensor(buf312, (1568, 2048), (2048, 1), 0); del buf312  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf423, permute_374, out=buf424)
    del permute_374
    buf427 = reinterpret_tensor(buf424, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf424  # reuse
    cpp_fused_gelu_gelu_backward_98(c_void_p(buf427.data_ptr()), c_void_p(addmm_28.data_ptr()))
    del addmm_28
    buf428 = reinterpret_tensor(buf422, (1568, 512), (512, 1), 0); del buf422  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf427, (1568, 2048), (2048, 1), 0), permute_378, out=buf428)
    del permute_378
    buf431 = buf413; del buf413  # reuse
    buf432 = buf412; del buf412  # reuse
    buf435 = reinterpret_tensor(buf332, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf332  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_99(c_void_p(buf428.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(getitem_35.data_ptr()), c_void_p(rsqrt_17.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf435.data_ptr()))
    del primals_49
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf436 = aten.convolution_backward(buf435, add_61, primals_209, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_61
    del buf435
    del primals_209
    buf437 = buf436[0]
    buf383 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf402 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf421 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf440 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_100(c_void_p(buf382.data_ptr()), c_void_p(addmm_33.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(addmm_31.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(addmm_29.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(addmm_27.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf440.data_ptr()))
    del addmm_27
    del addmm_29
    del addmm_31
    del addmm_33
    buf387 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf385, (512, 1568), (1, 512), 0), view_82, out=buf387)
    del view_82
    buf388 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_101(c_void_p(buf385.data_ptr()), c_void_p(buf388.data_ptr()))
    del buf385
    buf391 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf389, (2048, 1568), (1, 2048), 0), view_80, out=buf391)
    del view_80
    buf392 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf395 = empty((512, ), device='cpu', dtype=torch.float32)
    buf396 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_102(c_void_p(buf389.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(getitem_39.data_ptr()), c_void_p(rsqrt_19.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()))
    del buf390
    del convolution_19
    del getitem_39
    del rsqrt_19
    buf400 = buf398[1]
    buf401 = buf398[2]
    del buf398
    buf406 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf404, (512, 1568), (1, 512), 0), view_77, out=buf406)
    del view_77
    buf407 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_103(c_void_p(buf404.data_ptr()), c_void_p(buf407.data_ptr()))
    del buf404
    buf410 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf408, (2048, 1568), (1, 2048), 0), view_75, out=buf410)
    del view_75
    buf411 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf414 = empty((512, ), device='cpu', dtype=torch.float32)
    buf415 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_104(c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(getitem_37.data_ptr()), c_void_p(rsqrt_18.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()))
    del convolution_18
    del getitem_37
    del rsqrt_18
    buf419 = buf417[1]
    buf420 = buf417[2]
    del buf417
    buf425 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf423, (512, 1568), (1, 512), 0), view_72, out=buf425)
    del view_72
    buf426 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_105(c_void_p(buf423.data_ptr()), c_void_p(buf426.data_ptr()))
    buf429 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf427, (2048, 1568), (1, 2048), 0), view_70, out=buf429)
    del view_70
    buf430 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf433 = empty((512, ), device='cpu', dtype=torch.float32)
    buf434 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_106(c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(getitem_35.data_ptr()), c_void_p(rsqrt_17.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf434.data_ptr()))
    del convolution_17
    del getitem_35
    del rsqrt_17
    buf438 = buf436[1]
    buf439 = buf436[2]
    del buf436
    buf441 = reinterpret_tensor(buf428, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf428  # reuse
    buf442 = buf423; del buf423  # reuse
    cpp_fused_add_mul_view_107(c_void_p(buf382.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()))
    del primals_48
    buf443 = reinterpret_tensor(buf427, (1568, 2048), (2048, 1), 0); del buf427  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf442, permute_384, out=buf443)
    del permute_384
    buf444 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf442, (512, 1568), (1, 512), 0), view_67, out=buf444)
    del view_67
    buf445 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf446 = reinterpret_tensor(buf443, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf443  # reuse
    cpp_fused_gelu_gelu_backward_sum_108(c_void_p(buf446.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(addmm_26.data_ptr()), c_void_p(buf445.data_ptr()))
    del addmm_26
    buf447 = buf442; del buf442  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf446, (1568, 2048), (2048, 1), 0), permute_388, out=buf447)
    del permute_388
    buf448 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf446, (2048, 1568), (1, 2048), 0), view_65, out=buf448)
    del view_65
    buf449 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf450 = buf432; del buf432  # reuse
    buf451 = buf431; del buf431  # reuse
    buf452 = empty((512, ), device='cpu', dtype=torch.float32)
    buf453 = empty((512, ), device='cpu', dtype=torch.float32)
    buf454 = reinterpret_tensor(buf441, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf441  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_sum_109(c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(getitem_33.data_ptr()), c_void_p(rsqrt_16.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf454.data_ptr()))
    del convolution_16
    del getitem_33
    del primals_46
    del rsqrt_16
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf455 = aten.convolution_backward(buf454, add_57, primals_203, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_57
    del primals_203
    buf456 = buf455[0]
    buf457 = buf455[1]
    buf458 = buf455[2]
    del buf455
    buf459 = buf382; del buf382  # reuse
    buf461 = reinterpret_tensor(buf454, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf454  # reuse
    buf462 = buf447; del buf447  # reuse
    cpp_fused_add_mul_view_110(c_void_p(buf459.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()))
    del primals_45
    buf463 = reinterpret_tensor(buf446, (1568, 2048), (2048, 1), 0); del buf446  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf462, permute_394, out=buf463)
    del permute_394
    buf466 = reinterpret_tensor(buf463, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf463  # reuse
    cpp_fused_gelu_gelu_backward_111(c_void_p(buf466.data_ptr()), c_void_p(addmm_24.data_ptr()))
    del addmm_24
    buf467 = reinterpret_tensor(buf461, (1568, 512), (512, 1), 0); del buf461  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf466, (1568, 2048), (2048, 1), 0), permute_398, out=buf467)
    del permute_398
    buf470 = buf451; del buf451  # reuse
    buf471 = buf450; del buf450  # reuse
    buf474 = reinterpret_tensor(buf456, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf456  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_112(c_void_p(buf467.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(getitem_31.data_ptr()), c_void_p(rsqrt_15.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf474.data_ptr()))
    del primals_43
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf475 = aten.convolution_backward(buf474, add_53, primals_197, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_53
    del primals_197
    buf476 = buf475[0]
    buf480 = reinterpret_tensor(buf474, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf474  # reuse
    buf481 = reinterpret_tensor(buf437, (1568, 512), (512, 1), 0); del buf437  # reuse
    cpp_fused_add_mul_view_113(c_void_p(buf459.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf481.data_ptr()))
    del primals_42
    buf482 = reinterpret_tensor(buf408, (1568, 2048), (2048, 1), 0); del buf408  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf481, permute_404, out=buf482)
    del permute_404
    buf485 = reinterpret_tensor(buf482, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf482  # reuse
    cpp_fused_gelu_gelu_backward_114(c_void_p(buf485.data_ptr()), c_void_p(addmm_22.data_ptr()))
    del addmm_22
    buf486 = reinterpret_tensor(buf480, (1568, 512), (512, 1), 0); del buf480  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf485, (1568, 2048), (2048, 1), 0), permute_408, out=buf486)
    del permute_408
    buf489 = buf471; del buf471  # reuse
    buf490 = buf470; del buf470  # reuse
    buf493 = reinterpret_tensor(buf418, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf418  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_115(c_void_p(buf486.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(getitem_29.data_ptr()), c_void_p(rsqrt_14.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf493.data_ptr()))
    del primals_40
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf494 = aten.convolution_backward(buf493, add_49, primals_191, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_49
    del primals_191
    buf495 = buf494[0]
    buf499 = reinterpret_tensor(buf493, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf493  # reuse
    buf500 = reinterpret_tensor(buf399, (1568, 512), (512, 1), 0); del buf399  # reuse
    cpp_fused_add_mul_view_116(c_void_p(buf459.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()))
    del primals_39
    buf501 = reinterpret_tensor(buf389, (1568, 2048), (2048, 1), 0); del buf389  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf500, permute_414, out=buf501)
    del permute_414
    buf504 = reinterpret_tensor(buf501, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf501  # reuse
    cpp_fused_gelu_gelu_backward_117(c_void_p(buf504.data_ptr()), c_void_p(addmm_20.data_ptr()))
    del addmm_20
    buf505 = reinterpret_tensor(buf499, (1568, 512), (512, 1), 0); del buf499  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf504, (1568, 2048), (2048, 1), 0), permute_418, out=buf505)
    del permute_418
    buf508 = buf490; del buf490  # reuse
    buf509 = buf489; del buf489  # reuse
    buf512 = reinterpret_tensor(buf409, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf409  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_118(c_void_p(buf505.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(getitem_27.data_ptr()), c_void_p(rsqrt_13.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf512.data_ptr()))
    del primals_37
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf513 = aten.convolution_backward(buf512, add_45, primals_185, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_45
    del buf512
    del primals_185
    buf514 = buf513[0]
    buf460 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf479 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf498 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf517 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_119(c_void_p(buf459.data_ptr()), c_void_p(addmm_25.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(addmm_23.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(addmm_21.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(addmm_19.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf517.data_ptr()))
    del addmm_19
    del addmm_21
    del addmm_23
    del addmm_25
    buf464 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf462, (512, 1568), (1, 512), 0), view_62, out=buf464)
    del view_62
    buf465 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_120(c_void_p(buf462.data_ptr()), c_void_p(buf465.data_ptr()))
    del buf462
    buf468 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf466, (2048, 1568), (1, 2048), 0), view_60, out=buf468)
    del view_60
    buf469 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf472 = empty((512, ), device='cpu', dtype=torch.float32)
    buf473 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_121(c_void_p(buf466.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(getitem_31.data_ptr()), c_void_p(rsqrt_15.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf473.data_ptr()))
    del buf467
    del convolution_15
    del getitem_31
    del rsqrt_15
    buf477 = buf475[1]
    buf478 = buf475[2]
    del buf475
    buf483 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf481, (512, 1568), (1, 512), 0), view_57, out=buf483)
    del view_57
    buf484 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_122(c_void_p(buf481.data_ptr()), c_void_p(buf484.data_ptr()))
    del buf481
    buf487 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf485, (2048, 1568), (1, 2048), 0), view_55, out=buf487)
    del view_55
    buf488 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf491 = empty((512, ), device='cpu', dtype=torch.float32)
    buf492 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_123(c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(getitem_29.data_ptr()), c_void_p(rsqrt_14.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf492.data_ptr()))
    del buf486
    del convolution_14
    del getitem_29
    del rsqrt_14
    buf496 = buf494[1]
    buf497 = buf494[2]
    del buf494
    buf502 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf500, (512, 1568), (1, 512), 0), view_52, out=buf502)
    del view_52
    buf503 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_124(c_void_p(buf500.data_ptr()), c_void_p(buf503.data_ptr()))
    buf506 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf504, (2048, 1568), (1, 2048), 0), view_50, out=buf506)
    del view_50
    buf507 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf510 = empty((512, ), device='cpu', dtype=torch.float32)
    buf511 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_125(c_void_p(buf504.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(getitem_27.data_ptr()), c_void_p(rsqrt_13.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(buf511.data_ptr()))
    del convolution_13
    del getitem_27
    del rsqrt_13
    buf515 = buf513[1]
    buf516 = buf513[2]
    del buf513
    buf518 = reinterpret_tensor(buf505, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf505  # reuse
    buf519 = buf500; del buf500  # reuse
    cpp_fused_add_mul_view_126(c_void_p(buf459.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf519.data_ptr()))
    del primals_36
    buf520 = reinterpret_tensor(buf504, (1568, 2048), (2048, 1), 0); del buf504  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf519, permute_424, out=buf520)
    del permute_424
    buf521 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf519, (512, 1568), (1, 512), 0), view_47, out=buf521)
    del view_47
    buf522 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf523 = reinterpret_tensor(buf520, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf520  # reuse
    cpp_fused_gelu_gelu_backward_sum_127(c_void_p(buf523.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(addmm_18.data_ptr()), c_void_p(buf522.data_ptr()))
    del addmm_18
    buf524 = buf519; del buf519  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf523, (1568, 2048), (2048, 1), 0), permute_428, out=buf524)
    del permute_428
    buf525 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf523, (2048, 1568), (1, 2048), 0), view_45, out=buf525)
    del view_45
    buf526 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf527 = buf509; del buf509  # reuse
    buf528 = buf508; del buf508  # reuse
    buf529 = empty((512, ), device='cpu', dtype=torch.float32)
    buf530 = empty((512, ), device='cpu', dtype=torch.float32)
    buf531 = reinterpret_tensor(buf518, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf518  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_sum_128(c_void_p(buf523.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(getitem_25.data_ptr()), c_void_p(rsqrt_12.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf531.data_ptr()))
    del convolution_12
    del getitem_25
    del primals_34
    del rsqrt_12
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf532 = aten.convolution_backward(buf531, add_41, primals_179, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_41
    del primals_179
    buf533 = buf532[0]
    buf534 = buf532[1]
    buf535 = buf532[2]
    del buf532
    buf536 = buf459; del buf459  # reuse
    buf538 = reinterpret_tensor(buf531, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf531  # reuse
    buf539 = buf524; del buf524  # reuse
    cpp_fused_add_mul_view_129(c_void_p(buf536.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf539.data_ptr()))
    del buf476
    del primals_33
    buf540 = reinterpret_tensor(buf523, (1568, 2048), (2048, 1), 0); del buf523  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf539, permute_434, out=buf540)
    del permute_434
    buf543 = reinterpret_tensor(buf540, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf540  # reuse
    cpp_fused_gelu_gelu_backward_130(c_void_p(buf543.data_ptr()), c_void_p(addmm_16.data_ptr()))
    del addmm_16
    buf544 = reinterpret_tensor(buf538, (1568, 512), (512, 1), 0); del buf538  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf543, (1568, 2048), (2048, 1), 0), permute_438, out=buf544)
    del permute_438
    buf547 = buf528; del buf528  # reuse
    buf548 = buf527; del buf527  # reuse
    buf551 = reinterpret_tensor(buf533, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf533  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_131(c_void_p(buf544.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(getitem_23.data_ptr()), c_void_p(rsqrt_11.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf551.data_ptr()))
    del primals_31
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf552 = aten.convolution_backward(buf551, add_37, primals_173, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_37
    del primals_173
    buf553 = buf552[0]
    buf557 = reinterpret_tensor(buf551, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf551  # reuse
    buf558 = reinterpret_tensor(buf514, (1568, 512), (512, 1), 0); del buf514  # reuse
    cpp_fused_add_mul_view_132(c_void_p(buf536.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf558.data_ptr()))
    del primals_30
    buf559 = reinterpret_tensor(buf485, (1568, 2048), (2048, 1), 0); del buf485  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf558, permute_444, out=buf559)
    del permute_444
    buf562 = reinterpret_tensor(buf559, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf559  # reuse
    cpp_fused_gelu_gelu_backward_133(c_void_p(buf562.data_ptr()), c_void_p(addmm_14.data_ptr()))
    del addmm_14
    buf563 = reinterpret_tensor(buf557, (1568, 512), (512, 1), 0); del buf557  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf562, (1568, 2048), (2048, 1), 0), permute_448, out=buf563)
    del permute_448
    buf566 = buf548; del buf548  # reuse
    buf567 = buf547; del buf547  # reuse
    buf570 = reinterpret_tensor(buf495, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf495  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_134(c_void_p(buf563.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(getitem_21.data_ptr()), c_void_p(rsqrt_10.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf570.data_ptr()))
    del primals_28
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf571 = aten.convolution_backward(buf570, add_33, primals_167, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del add_33
    del buf570
    del primals_167
    buf572 = buf571[0]
    buf537 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf556 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    buf575 = empty((1, 512, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_135(c_void_p(buf536.data_ptr()), c_void_p(addmm_17.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(addmm_15.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(addmm_13.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf575.data_ptr()))
    del addmm_13
    del addmm_15
    del addmm_17
    buf541 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf539, (512, 1568), (1, 512), 0), view_42, out=buf541)
    del view_42
    buf542 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_136(c_void_p(buf539.data_ptr()), c_void_p(buf542.data_ptr()))
    del buf539
    buf545 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf543, (2048, 1568), (1, 2048), 0), view_40, out=buf545)
    del view_40
    buf546 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf549 = empty((512, ), device='cpu', dtype=torch.float32)
    buf550 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_137(c_void_p(buf543.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(getitem_23.data_ptr()), c_void_p(rsqrt_11.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf550.data_ptr()))
    del buf544
    del convolution_11
    del getitem_23
    del rsqrt_11
    buf554 = buf552[1]
    buf555 = buf552[2]
    del buf552
    buf560 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf558, (512, 1568), (1, 512), 0), view_37, out=buf560)
    del view_37
    buf561 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_sum_138(c_void_p(buf558.data_ptr()), c_void_p(buf561.data_ptr()))
    buf564 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf562, (2048, 1568), (1, 2048), 0), view_35, out=buf564)
    del view_35
    buf565 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf568 = empty((512, ), device='cpu', dtype=torch.float32)
    buf569 = empty((512, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_139(c_void_p(buf562.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(getitem_21.data_ptr()), c_void_p(rsqrt_10.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf569.data_ptr()))
    del convolution_10
    del getitem_21
    del rsqrt_10
    buf573 = buf571[1]
    buf574 = buf571[2]
    del buf571
    buf576 = reinterpret_tensor(buf563, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf563  # reuse
    buf577 = buf558; del buf558  # reuse
    cpp_fused_add_mul_view_140(c_void_p(buf536.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf577.data_ptr()))
    del primals_27
    buf578 = reinterpret_tensor(buf562, (1568, 2048), (2048, 1), 0); del buf562  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf577, permute_454, out=buf578)
    del permute_454
    buf579 = empty((512, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf577, (512, 1568), (1, 512), 0), view_32, out=buf579)
    del view_32
    buf580 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf581 = reinterpret_tensor(buf578, (8, 14, 14, 2048), (401408, 28672, 2048, 1), 0); del buf578  # reuse
    cpp_fused_gelu_gelu_backward_sum_141(c_void_p(buf581.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(addmm_12.data_ptr()), c_void_p(buf580.data_ptr()))
    del addmm_12
    buf582 = buf577; del buf577  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf581, (1568, 2048), (2048, 1), 0), permute_458, out=buf582)
    del permute_458
    buf583 = empty((2048, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf581, (2048, 1568), (1, 2048), 0), view_30, out=buf583)
    del view_30
    buf584 = empty((1, 2048), device='cpu', dtype=torch.float32)
    buf585 = buf567; del buf567  # reuse
    buf586 = buf566; del buf566  # reuse
    buf587 = empty((512, ), device='cpu', dtype=torch.float32)
    buf588 = empty((512, ), device='cpu', dtype=torch.float32)
    buf589 = reinterpret_tensor(buf576, (8, 512, 14, 14), (100352, 1, 512, 7168), 0); del buf576  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_sum_142(c_void_p(buf581.data_ptr()), c_void_p(buf582.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(getitem_19.data_ptr()), c_void_p(rsqrt_9.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(buf589.data_ptr()))
    del buf582
    del buf585
    del buf586
    del convolution_9
    del getitem_19
    del primals_25
    del rsqrt_9
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf590 = aten.convolution_backward(buf589, convolution_8, primals_161, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True])
    del convolution_8
    del primals_161
    buf591 = buf590[0]
    buf592 = buf590[1]
    buf593 = buf590[2]
    del buf590
    buf594 = buf589; del buf589  # reuse
    cpp_fused_add_convolution_backward_143(c_void_p(buf536.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf594.data_ptr()))
    del buf536
    del buf553
    del buf572
    del buf591
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]
    buf595 = aten.convolution_backward(buf594, permute_29, primals_159, [512], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf594
    del permute_29
    del primals_159
    buf596 = buf595[0]
    buf597 = buf595[1]
    buf598 = buf595[2]
    del buf595
    buf599 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf600 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf601 = empty((256, ), device='cpu', dtype=torch.float32)
    buf602 = empty((256, ), device='cpu', dtype=torch.float32)
    buf603 = reinterpret_tensor(buf596, (8, 28, 28, 256), (200704, 7168, 256, 1), 0); del buf596  # reuse
    buf605 = reinterpret_tensor(buf52, (8, 256, 28, 28), (200704, 784, 28, 1), 0); del buf52  # reuse
    buf606 = reinterpret_tensor(buf16, (6272, 256), (256, 1), 0); del buf16  # reuse
    cpp_fused_mul_native_layer_norm_backward_view_144(c_void_p(buf603.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(mul_40.data_ptr()), c_void_p(div_33.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf606.data_ptr()))
    del div_33
    del mul_40
    del primals_22
    del primals_23
    buf607 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf606, permute_466, out=buf607)
    del permute_466
    buf610 = reinterpret_tensor(buf607, (8, 28, 28, 1024), (802816, 28672, 1024, 1), 0); del buf607  # reuse
    cpp_fused_gelu_gelu_backward_145(c_void_p(buf610.data_ptr()), c_void_p(addmm_10.data_ptr()))
    del addmm_10
    buf611 = reinterpret_tensor(buf605, (6272, 256), (256, 1), 0); del buf605  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf610, (6272, 1024), (1024, 1), 0), permute_470, out=buf611)
    del permute_470
    buf614 = buf600; del buf600  # reuse
    buf615 = buf599; del buf599  # reuse
    buf618 = empty_strided((8, 256, 28, 28), (200704, 1, 256, 7168), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_146(c_void_p(buf611.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(getitem_15.data_ptr()), c_void_p(rsqrt_7.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf615.data_ptr()), c_void_p(buf618.data_ptr()))
    del primals_20
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf619 = aten.convolution_backward(buf618, add_23, primals_153, [256], [1, 1], [3, 3], [1, 1], False, [0, 0], 256, [True, True, True])
    del add_23
    del primals_153
    buf620 = buf619[0]
    buf624 = reinterpret_tensor(buf618, (8, 256, 28, 28), (200704, 784, 28, 1), 0); del buf618  # reuse
    buf625 = empty((6272, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_view_147(c_void_p(buf603.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf625.data_ptr()))
    del primals_19
    buf626 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf625, permute_476, out=buf626)
    del permute_476
    buf629 = reinterpret_tensor(buf626, (8, 28, 28, 1024), (802816, 28672, 1024, 1), 0); del buf626  # reuse
    cpp_fused_gelu_gelu_backward_148(c_void_p(buf629.data_ptr()), c_void_p(addmm_8.data_ptr()))
    del addmm_8
    buf630 = reinterpret_tensor(buf624, (6272, 256), (256, 1), 0); del buf624  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf629, (6272, 1024), (1024, 1), 0), permute_480, out=buf630)
    del permute_480
    buf633 = buf615; del buf615  # reuse
    buf634 = buf614; del buf614  # reuse
    buf637 = empty_strided((8, 256, 28, 28), (200704, 1, 256, 7168), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_149(c_void_p(buf630.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(getitem_13.data_ptr()), c_void_p(rsqrt_6.data_ptr()), c_void_p(buf633.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(buf637.data_ptr()))
    del primals_17
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf638 = aten.convolution_backward(buf637, add_19, primals_147, [256], [1, 1], [3, 3], [1, 1], False, [0, 0], 256, [True, True, True])
    del add_19
    del buf637
    del primals_147
    buf639 = buf638[0]
    buf604 = empty((1, 256, 1, 1), device='cpu', dtype=torch.float32)
    buf623 = empty((1, 256, 1, 1), device='cpu', dtype=torch.float32)
    buf642 = empty((1, 256, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_sum_150(c_void_p(buf603.data_ptr()), c_void_p(addmm_11.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(addmm_9.data_ptr()), c_void_p(buf639.data_ptr()), c_void_p(addmm_7.data_ptr()), c_void_p(buf604.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(buf642.data_ptr()))
    del addmm_11
    del addmm_7
    del addmm_9
    buf608 = empty((256, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf606, (256, 6272), (1, 256), 0), view_27, out=buf608)
    del view_27
    buf609 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_151(c_void_p(buf606.data_ptr()), c_void_p(buf609.data_ptr()))
    del buf606
    buf612 = empty((1024, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf610, (1024, 6272), (1, 1024), 0), view_25, out=buf612)
    del view_25
    buf613 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf616 = empty((256, ), device='cpu', dtype=torch.float32)
    buf617 = empty((256, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_152(c_void_p(buf610.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(getitem_15.data_ptr()), c_void_p(rsqrt_7.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(buf617.data_ptr()))
    del buf610
    del buf611
    del convolution_7
    del getitem_15
    del rsqrt_7
    buf621 = buf619[1]
    buf622 = buf619[2]
    del buf619
    buf627 = empty((256, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf625, (256, 6272), (1, 256), 0), view_22, out=buf627)
    del view_22
    buf628 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_153(c_void_p(buf625.data_ptr()), c_void_p(buf628.data_ptr()))
    buf631 = empty((1024, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf629, (1024, 6272), (1, 1024), 0), view_20, out=buf631)
    del view_20
    buf632 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf635 = empty((256, ), device='cpu', dtype=torch.float32)
    buf636 = empty((256, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_154(c_void_p(buf629.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(getitem_13.data_ptr()), c_void_p(rsqrt_6.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(buf635.data_ptr()), c_void_p(buf636.data_ptr()))
    del convolution_6
    del getitem_13
    del rsqrt_6
    buf640 = buf638[1]
    buf641 = buf638[2]
    del buf638
    buf643 = reinterpret_tensor(buf630, (8, 256, 28, 28), (200704, 784, 28, 1), 0); del buf630  # reuse
    buf644 = buf625; del buf625  # reuse
    cpp_fused_add_mul_view_155(c_void_p(buf603.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(buf639.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf643.data_ptr()), c_void_p(buf644.data_ptr()))
    del primals_16
    buf645 = reinterpret_tensor(buf629, (6272, 1024), (1024, 1), 0); del buf629  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf644, permute_486, out=buf645)
    del permute_486
    buf646 = empty((256, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf644, (256, 6272), (1, 256), 0), view_17, out=buf646)
    del view_17
    buf647 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf648 = reinterpret_tensor(buf645, (8, 28, 28, 1024), (802816, 28672, 1024, 1), 0); del buf645  # reuse
    cpp_fused_gelu_gelu_backward_sum_156(c_void_p(buf648.data_ptr()), c_void_p(buf644.data_ptr()), c_void_p(addmm_6.data_ptr()), c_void_p(buf647.data_ptr()))
    del addmm_6
    buf649 = buf644; del buf644  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf648, (6272, 1024), (1024, 1), 0), permute_490, out=buf649)
    del permute_490
    buf650 = empty((1024, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf648, (1024, 6272), (1, 1024), 0), view_15, out=buf650)
    del view_15
    buf651 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf652 = buf634; del buf634  # reuse
    buf653 = buf633; del buf633  # reuse
    buf654 = empty((256, ), device='cpu', dtype=torch.float32)
    buf655 = empty((256, ), device='cpu', dtype=torch.float32)
    buf656 = reinterpret_tensor(buf643, (8, 256, 28, 28), (200704, 1, 256, 7168), 0); del buf643  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_sum_157(c_void_p(buf648.data_ptr()), c_void_p(buf649.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(getitem_11.data_ptr()), c_void_p(rsqrt_5.data_ptr()), c_void_p(buf651.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(buf653.data_ptr()), c_void_p(buf654.data_ptr()), c_void_p(buf655.data_ptr()), c_void_p(buf656.data_ptr()))
    del buf648
    del buf649
    del buf652
    del buf653
    del convolution_5
    del getitem_11
    del primals_14
    del rsqrt_5
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf657 = aten.convolution_backward(buf656, convolution_4, primals_141, [256], [1, 1], [3, 3], [1, 1], False, [0, 0], 256, [True, True, True])
    del convolution_4
    del primals_141
    buf658 = buf657[0]
    buf659 = buf657[1]
    buf660 = buf657[2]
    del buf657
    buf661 = buf656; del buf656  # reuse
    cpp_fused_add_convolution_backward_158(c_void_p(buf603.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(buf639.data_ptr()), c_void_p(buf658.data_ptr()), c_void_p(buf661.data_ptr()))
    del buf603
    del buf620
    del buf639
    del buf658
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]
    buf662 = aten.convolution_backward(buf661, permute_15, primals_139, [256], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf661
    del permute_15
    del primals_139
    buf663 = buf662[0]
    buf664 = buf662[1]
    buf665 = buf662[2]
    del buf662
    buf666 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cpu', dtype=torch.float32)
    buf667 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cpu', dtype=torch.float32)
    buf668 = empty((128, ), device='cpu', dtype=torch.float32)
    buf669 = empty((128, ), device='cpu', dtype=torch.float32)
    buf670 = reinterpret_tensor(buf663, (8, 56, 56, 128), (401408, 7168, 128, 1), 0); del buf663  # reuse
    buf672 = reinterpret_tensor(buf581, (8, 128, 56, 56), (401408, 3136, 56, 1), 0); del buf581  # reuse
    buf673 = reinterpret_tensor(buf543, (25088, 128), (128, 1), 0); del buf543  # reuse
    cpp_fused_mul_native_layer_norm_backward_view_159(c_void_p(buf670.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(mul_20.data_ptr()), c_void_p(div_37.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf666.data_ptr()), c_void_p(buf667.data_ptr()), c_void_p(buf668.data_ptr()), c_void_p(buf669.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf673.data_ptr()))
    del div_37
    del mul_20
    del primals_11
    del primals_12
    buf674 = empty((25088, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf673, permute_498, out=buf674)
    del permute_498
    buf677 = reinterpret_tensor(buf674, (8, 56, 56, 512), (1605632, 28672, 512, 1), 0); del buf674  # reuse
    cpp_fused_gelu_gelu_backward_160(c_void_p(buf677.data_ptr()), c_void_p(addmm_4.data_ptr()))
    del addmm_4
    buf678 = reinterpret_tensor(buf672, (25088, 128), (128, 1), 0); del buf672  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf677, (25088, 512), (512, 1), 0), permute_502, out=buf678)
    del permute_502
    buf681 = buf667; del buf667  # reuse
    buf682 = buf666; del buf666  # reuse
    buf685 = reinterpret_tensor(buf466, (8, 128, 56, 56), (401408, 1, 128, 7168), 0); del buf466  # reuse
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_161(c_void_p(buf678.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(getitem_7.data_ptr()), c_void_p(rsqrt_3.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(buf682.data_ptr()), c_void_p(buf685.data_ptr()))
    del primals_9
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf686 = aten.convolution_backward(buf685, add_9, primals_133, [128], [1, 1], [3, 3], [1, 1], False, [0, 0], 128, [True, True, True])
    del add_9
    del primals_133
    buf687 = buf686[0]
    buf691 = reinterpret_tensor(buf685, (8, 128, 56, 56), (401408, 3136, 56, 1), 0); del buf685  # reuse
    buf692 = empty((25088, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_view_162(c_void_p(buf670.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf691.data_ptr()), c_void_p(buf692.data_ptr()))
    del primals_8
    buf693 = empty((25088, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf692, permute_508, out=buf693)
    del permute_508
    buf696 = reinterpret_tensor(buf693, (8, 56, 56, 512), (1605632, 28672, 512, 1), 0); del buf693  # reuse
    cpp_fused_gelu_gelu_backward_163(c_void_p(buf696.data_ptr()), c_void_p(addmm_2.data_ptr()))
    del addmm_2
    buf697 = reinterpret_tensor(buf691, (25088, 128), (128, 1), 0); del buf691  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf696, (25088, 512), (512, 1), 0), permute_512, out=buf697)
    del permute_512
    buf700 = buf682; del buf682  # reuse
    buf701 = buf681; del buf681  # reuse
    buf704 = empty_strided((8, 128, 56, 56), (401408, 1, 128, 7168), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_164(c_void_p(buf697.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(getitem_5.data_ptr()), c_void_p(rsqrt_2.data_ptr()), c_void_p(buf700.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf704.data_ptr()))
    del primals_6
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf705 = aten.convolution_backward(buf704, add_5, primals_127, [128], [1, 1], [3, 3], [1, 1], False, [0, 0], 128, [True, True, True])
    del add_5
    del primals_127
    buf706 = buf705[0]
    buf710 = reinterpret_tensor(buf704, (8, 128, 56, 56), (401408, 3136, 56, 1), 0); del buf704  # reuse
    buf711 = empty((25088, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_view_165(c_void_p(buf670.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf710.data_ptr()), c_void_p(buf711.data_ptr()))
    del primals_5
    buf712 = empty((25088, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf711, permute_518, out=buf712)
    del permute_518
    buf715 = reinterpret_tensor(buf712, (8, 56, 56, 512), (1605632, 28672, 512, 1), 0); del buf712  # reuse
    cpp_fused_gelu_gelu_backward_166(c_void_p(buf715.data_ptr()), c_void_p(addmm.data_ptr()))
    del addmm
    buf716 = reinterpret_tensor(buf710, (25088, 128), (128, 1), 0); del buf710  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf715, (25088, 512), (512, 1), 0), permute_522, out=buf716)
    del permute_522
    buf719 = buf701; del buf701  # reuse
    buf720 = buf700; del buf700  # reuse
    buf723 = empty_strided((8, 128, 56, 56), (401408, 1, 128, 7168), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_layer_norm_native_layer_norm_backward_167(c_void_p(buf716.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(getitem_3.data_ptr()), c_void_p(rsqrt_1.data_ptr()), c_void_p(buf719.data_ptr()), c_void_p(buf720.data_ptr()), c_void_p(buf723.data_ptr()))
    del primals_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf724 = aten.convolution_backward(buf723, permute_1, primals_121, [128], [1, 1], [3, 3], [1, 1], False, [0, 0], 128, [True, True, True])
    del buf723
    del permute_1
    del primals_121
    buf725 = buf724[0]
    buf671 = empty((1, 128, 1, 1), device='cpu', dtype=torch.float32)
    buf690 = empty((1, 128, 1, 1), device='cpu', dtype=torch.float32)
    buf709 = empty((1, 128, 1, 1), device='cpu', dtype=torch.float32)
    buf732 = empty((128, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_168(c_void_p(buf670.data_ptr()), c_void_p(addmm_5.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(addmm_3.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(addmm_1.data_ptr()), c_void_p(buf725.data_ptr()), c_void_p(buf671.data_ptr()), c_void_p(buf690.data_ptr()), c_void_p(buf709.data_ptr()), c_void_p(buf732.data_ptr()))
    del addmm_1
    del addmm_3
    del addmm_5
    buf675 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf673, (128, 25088), (1, 128), 0), view_12, out=buf675)
    del view_12
    buf676 = empty((1, 128), device='cpu', dtype=torch.float32)
    cpp_fused_sum_169(c_void_p(buf673.data_ptr()), c_void_p(buf676.data_ptr()))
    del buf673
    buf679 = empty((512, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf677, (512, 25088), (1, 512), 0), view_10, out=buf679)
    del view_10
    buf680 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf683 = empty((128, ), device='cpu', dtype=torch.float32)
    buf684 = empty((128, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_170(c_void_p(buf677.data_ptr()), c_void_p(buf678.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(getitem_7.data_ptr()), c_void_p(rsqrt_3.data_ptr()), c_void_p(buf680.data_ptr()), c_void_p(buf683.data_ptr()), c_void_p(buf684.data_ptr()))
    del buf677
    del buf678
    del convolution_3
    del getitem_7
    del rsqrt_3
    buf688 = buf686[1]
    buf689 = buf686[2]
    del buf686
    buf694 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf692, (128, 25088), (1, 128), 0), view_7, out=buf694)
    del view_7
    buf695 = empty((1, 128), device='cpu', dtype=torch.float32)
    cpp_fused_sum_171(c_void_p(buf692.data_ptr()), c_void_p(buf695.data_ptr()))
    del buf692
    buf698 = empty((512, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf696, (512, 25088), (1, 512), 0), view_5, out=buf698)
    del view_5
    buf699 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf702 = empty((128, ), device='cpu', dtype=torch.float32)
    buf703 = empty((128, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_172(c_void_p(buf696.data_ptr()), c_void_p(buf697.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(getitem_5.data_ptr()), c_void_p(rsqrt_2.data_ptr()), c_void_p(buf699.data_ptr()), c_void_p(buf702.data_ptr()), c_void_p(buf703.data_ptr()))
    del buf696
    del buf697
    del convolution_2
    del getitem_5
    del rsqrt_2
    buf707 = buf705[1]
    buf708 = buf705[2]
    del buf705
    buf713 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf711, (128, 25088), (1, 128), 0), view_2, out=buf713)
    del view_2
    buf714 = empty((1, 128), device='cpu', dtype=torch.float32)
    cpp_fused_sum_173(c_void_p(buf711.data_ptr()), c_void_p(buf714.data_ptr()))
    buf717 = empty((512, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf715, (512, 25088), (1, 512), 0), view, out=buf717)
    del view
    buf718 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf721 = empty((128, ), device='cpu', dtype=torch.float32)
    buf722 = empty((128, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_sum_174(c_void_p(buf715.data_ptr()), c_void_p(buf716.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(getitem_3.data_ptr()), c_void_p(rsqrt_1.data_ptr()), c_void_p(buf718.data_ptr()), c_void_p(buf721.data_ptr()), c_void_p(buf722.data_ptr()))
    del buf715
    del convolution_1
    del getitem_3
    del rsqrt_1
    buf726 = buf724[1]
    buf727 = buf724[2]
    del buf724
    buf728 = reinterpret_tensor(buf716, (8, 56, 56, 128), (401408, 7168, 128, 1), 0); del buf716  # reuse
    buf729 = buf720; del buf720  # reuse
    buf730 = buf719; del buf719  # reuse
    buf731 = empty((128, ), device='cpu', dtype=torch.float32)
    buf733 = reinterpret_tensor(buf711, (8, 128, 56, 56), (401408, 1, 128, 7168), 0); del buf711  # reuse
    cpp_fused_convolution_backward_native_layer_norm_backward_175(c_void_p(buf670.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(buf725.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(mul.data_ptr()), c_void_p(div_41.data_ptr()), c_void_p(buf728.data_ptr()), c_void_p(buf729.data_ptr()), c_void_p(buf730.data_ptr()), c_void_p(buf731.data_ptr()), c_void_p(buf733.data_ptr()))
    del buf670
    del buf687
    del buf706
    del buf725
    del buf728
    del buf729
    del buf730
    del div_41
    del mul
    del primals_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf734 = aten.convolution_backward(buf733, primals_345, primals_119, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf733
    del primals_119
    del primals_345
    buf735 = buf734[1]
    buf736 = buf734[2]
    return (buf731, buf732, buf721, buf722, reinterpret_tensor(buf709, (128, ), (1, ), 0), buf702, buf703, reinterpret_tensor(buf690, (128, ), (1, ), 0), buf683, buf684, reinterpret_tensor(buf671, (128, ), (1, ), 0), buf668, buf669, buf654, buf655, reinterpret_tensor(buf642, (256, ), (1, ), 0), buf635, buf636, reinterpret_tensor(buf623, (256, ), (1, ), 0), buf616, buf617, reinterpret_tensor(buf604, (256, ), (1, ), 0), buf601, buf602, buf587, buf588, reinterpret_tensor(buf575, (512, ), (1, ), 0), buf568, buf569, reinterpret_tensor(buf556, (512, ), (1, ), 0), buf549, buf550, reinterpret_tensor(buf537, (512, ), (1, ), 0), buf529, buf530, reinterpret_tensor(buf517, (512, ), (1, ), 0), buf510, buf511, reinterpret_tensor(buf498, (512, ), (1, ), 0), buf491, buf492, reinterpret_tensor(buf479, (512, ), (1, ), 0), buf472, buf473, reinterpret_tensor(buf460, (512, ), (1, ), 0), buf452, buf453, reinterpret_tensor(buf440, (512, ), (1, ), 0), buf433, buf434, reinterpret_tensor(buf421, (512, ), (1, ), 0), buf414, buf415, reinterpret_tensor(buf402, (512, ), (1, ), 0), buf395, buf396, reinterpret_tensor(buf383, (512, ), (1, ), 0), buf375, buf376, reinterpret_tensor(buf363, (512, ), (1, ), 0), buf356, buf357, reinterpret_tensor(buf344, (512, ), (1, ), 0), buf337, buf338, reinterpret_tensor(buf325, (512, ), (1, ), 0), buf318, buf319, reinterpret_tensor(buf306, (512, ), (1, ), 0), buf298, buf299, reinterpret_tensor(buf286, (512, ), (1, ), 0), buf279, buf280, reinterpret_tensor(buf267, (512, ), (1, ), 0), buf260, buf261, reinterpret_tensor(buf248, (512, ), (1, ), 0), buf241, buf242, reinterpret_tensor(buf229, (512, ), (1, ), 0), buf221, buf222, reinterpret_tensor(buf209, (512, ), (1, ), 0), buf202, buf203, reinterpret_tensor(buf190, (512, ), (1, ), 0), buf183, buf184, reinterpret_tensor(buf171, (512, ), (1, ), 0), buf164, buf165, reinterpret_tensor(buf152, (512, ), (1, ), 0), buf144, buf145, reinterpret_tensor(buf132, (512, ), (1, ), 0), buf125, buf126, reinterpret_tensor(buf113, (512, ), (1, ), 0), buf106, buf107, reinterpret_tensor(buf94, (512, ), (1, ), 0), buf87, buf88, reinterpret_tensor(buf75, (512, ), (1, ), 0), buf72, buf73, buf58, buf59, reinterpret_tensor(buf47, (1024, ), (1, ), 0), buf40, buf41, reinterpret_tensor(buf29, (1024, ), (1, ), 0), buf22, buf23, reinterpret_tensor(buf11, (1024, ), (1, ), 0), buf5, buf6, buf735, buf736, buf726, buf727, reinterpret_tensor(buf717, (512, 128), (128, 1), 0), reinterpret_tensor(buf718, (512, ), (1, ), 0), reinterpret_tensor(buf713, (128, 512), (512, 1), 0), reinterpret_tensor(buf714, (128, ), (1, ), 0), buf707, buf708, reinterpret_tensor(buf698, (512, 128), (128, 1), 0), reinterpret_tensor(buf699, (512, ), (1, ), 0), reinterpret_tensor(buf694, (128, 512), (512, 1), 0), reinterpret_tensor(buf695, (128, ), (1, ), 0), buf688, buf689, reinterpret_tensor(buf679, (512, 128), (128, 1), 0), reinterpret_tensor(buf680, (512, ), (1, ), 0), reinterpret_tensor(buf675, (128, 512), (512, 1), 0), reinterpret_tensor(buf676, (128, ), (1, ), 0), buf664, buf665, buf659, buf660, reinterpret_tensor(buf650, (1024, 256), (256, 1), 0), reinterpret_tensor(buf651, (1024, ), (1, ), 0), reinterpret_tensor(buf646, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf647, (256, ), (1, ), 0), buf640, buf641, reinterpret_tensor(buf631, (1024, 256), (256, 1), 0), reinterpret_tensor(buf632, (1024, ), (1, ), 0), reinterpret_tensor(buf627, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf628, (256, ), (1, ), 0), buf621, buf622, reinterpret_tensor(buf612, (1024, 256), (256, 1), 0), reinterpret_tensor(buf613, (1024, ), (1, ), 0), reinterpret_tensor(buf608, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf609, (256, ), (1, ), 0), buf597, buf598, buf592, buf593, reinterpret_tensor(buf583, (2048, 512), (512, 1), 0), reinterpret_tensor(buf584, (2048, ), (1, ), 0), reinterpret_tensor(buf579, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf580, (512, ), (1, ), 0), buf573, buf574, reinterpret_tensor(buf564, (2048, 512), (512, 1), 0), reinterpret_tensor(buf565, (2048, ), (1, ), 0), reinterpret_tensor(buf560, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf561, (512, ), (1, ), 0), buf554, buf555, reinterpret_tensor(buf545, (2048, 512), (512, 1), 0), reinterpret_tensor(buf546, (2048, ), (1, ), 0), reinterpret_tensor(buf541, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf542, (512, ), (1, ), 0), buf534, buf535, reinterpret_tensor(buf525, (2048, 512), (512, 1), 0), reinterpret_tensor(buf526, (2048, ), (1, ), 0), reinterpret_tensor(buf521, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf522, (512, ), (1, ), 0), buf515, buf516, reinterpret_tensor(buf506, (2048, 512), (512, 1), 0), reinterpret_tensor(buf507, (2048, ), (1, ), 0), reinterpret_tensor(buf502, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf503, (512, ), (1, ), 0), buf496, buf497, reinterpret_tensor(buf487, (2048, 512), (512, 1), 0), reinterpret_tensor(buf488, (2048, ), (1, ), 0), reinterpret_tensor(buf483, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf484, (512, ), (1, ), 0), buf477, buf478, reinterpret_tensor(buf468, (2048, 512), (512, 1), 0), reinterpret_tensor(buf469, (2048, ), (1, ), 0), reinterpret_tensor(buf464, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf465, (512, ), (1, ), 0), buf457, buf458, reinterpret_tensor(buf448, (2048, 512), (512, 1), 0), reinterpret_tensor(buf449, (2048, ), (1, ), 0), reinterpret_tensor(buf444, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf445, (512, ), (1, ), 0), buf438, buf439, reinterpret_tensor(buf429, (2048, 512), (512, 1), 0), reinterpret_tensor(buf430, (2048, ), (1, ), 0), reinterpret_tensor(buf425, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf426, (512, ), (1, ), 0), buf419, buf420, reinterpret_tensor(buf410, (2048, 512), (512, 1), 0), reinterpret_tensor(buf411, (2048, ), (1, ), 0), reinterpret_tensor(buf406, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf407, (512, ), (1, ), 0), buf400, buf401, reinterpret_tensor(buf391, (2048, 512), (512, 1), 0), reinterpret_tensor(buf392, (2048, ), (1, ), 0), reinterpret_tensor(buf387, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf388, (512, ), (1, ), 0), buf380, buf381, reinterpret_tensor(buf371, (2048, 512), (512, 1), 0), reinterpret_tensor(buf372, (2048, ), (1, ), 0), reinterpret_tensor(buf367, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf368, (512, ), (1, ), 0), buf361, buf362, reinterpret_tensor(buf352, (2048, 512), (512, 1), 0), reinterpret_tensor(buf353, (2048, ), (1, ), 0), reinterpret_tensor(buf348, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf349, (512, ), (1, ), 0), buf342, buf343, reinterpret_tensor(buf333, (2048, 512), (512, 1), 0), reinterpret_tensor(buf334, (2048, ), (1, ), 0), reinterpret_tensor(buf329, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf330, (512, ), (1, ), 0), buf323, buf324, reinterpret_tensor(buf314, (2048, 512), (512, 1), 0), reinterpret_tensor(buf315, (2048, ), (1, ), 0), reinterpret_tensor(buf310, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf311, (512, ), (1, ), 0), buf303, buf304, reinterpret_tensor(buf294, (2048, 512), (512, 1), 0), reinterpret_tensor(buf295, (2048, ), (1, ), 0), reinterpret_tensor(buf290, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf291, (512, ), (1, ), 0), buf284, buf285, reinterpret_tensor(buf275, (2048, 512), (512, 1), 0), reinterpret_tensor(buf276, (2048, ), (1, ), 0), reinterpret_tensor(buf271, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf272, (512, ), (1, ), 0), buf265, buf266, reinterpret_tensor(buf256, (2048, 512), (512, 1), 0), reinterpret_tensor(buf257, (2048, ), (1, ), 0), reinterpret_tensor(buf252, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf253, (512, ), (1, ), 0), buf246, buf247, reinterpret_tensor(buf237, (2048, 512), (512, 1), 0), reinterpret_tensor(buf238, (2048, ), (1, ), 0), reinterpret_tensor(buf233, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf234, (512, ), (1, ), 0), buf226, buf227, reinterpret_tensor(buf217, (2048, 512), (512, 1), 0), reinterpret_tensor(buf218, (2048, ), (1, ), 0), reinterpret_tensor(buf213, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf214, (512, ), (1, ), 0), buf207, buf208, reinterpret_tensor(buf198, (2048, 512), (512, 1), 0), reinterpret_tensor(buf199, (2048, ), (1, ), 0), reinterpret_tensor(buf194, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf195, (512, ), (1, ), 0), buf188, buf189, reinterpret_tensor(buf179, (2048, 512), (512, 1), 0), reinterpret_tensor(buf180, (2048, ), (1, ), 0), reinterpret_tensor(buf175, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf176, (512, ), (1, ), 0), buf169, buf170, reinterpret_tensor(buf160, (2048, 512), (512, 1), 0), reinterpret_tensor(buf161, (2048, ), (1, ), 0), reinterpret_tensor(buf156, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf157, (512, ), (1, ), 0), buf149, buf150, reinterpret_tensor(buf140, (2048, 512), (512, 1), 0), reinterpret_tensor(buf141, (2048, ), (1, ), 0), reinterpret_tensor(buf136, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf137, (512, ), (1, ), 0), buf130, buf131, reinterpret_tensor(buf121, (2048, 512), (512, 1), 0), reinterpret_tensor(buf122, (2048, ), (1, ), 0), reinterpret_tensor(buf117, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf118, (512, ), (1, ), 0), buf111, buf112, reinterpret_tensor(buf102, (2048, 512), (512, 1), 0), reinterpret_tensor(buf103, (2048, ), (1, ), 0), reinterpret_tensor(buf98, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf99, (512, ), (1, ), 0), buf92, buf93, reinterpret_tensor(buf83, (2048, 512), (512, 1), 0), reinterpret_tensor(buf84, (2048, ), (1, ), 0), reinterpret_tensor(buf79, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf80, (512, ), (1, ), 0), buf68, buf69, buf63, buf64, reinterpret_tensor(buf54, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf55, (4096, ), (1, ), 0), reinterpret_tensor(buf50, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf51, (1024, ), (1, ), 0), buf45, buf46, reinterpret_tensor(buf36, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf37, (4096, ), (1, ), 0), reinterpret_tensor(buf32, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf33, (1024, ), (1, ), 0), buf27, buf28, reinterpret_tensor(buf18, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf19, (4096, ), (1, ), 0), reinterpret_tensor(buf14, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf15, (1024, ), (1, ), 0), reinterpret_tensor(buf1, (1000, 1024), (1024, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((128, 3, 4, 4), (48, 1, 12, 3), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((256, 128, 2, 2), (512, 1, 256, 128), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((512, 256, 2, 2), (1024, 1, 512, 256), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_317 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_323 = rand_strided((1024, 512, 2, 2), (2048, 1, 1024, 512), device='cpu', dtype=torch.float32)
    primals_325 = rand_strided((1024, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_331 = rand_strided((1024, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_337 = rand_strided((1024, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_345 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    mul = rand_strided((8, 56, 56, 128), (401408, 1, 7168, 56), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cpu', dtype=torch.float32)
    rsqrt_1 = rand_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cpu', dtype=torch.float32)
    view = rand_strided((25088, 128), (128, 1), device='cpu', dtype=torch.float32)
    addmm = rand_strided((25088, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_2 = rand_strided((25088, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_1 = rand_strided((25088, 128), (128, 1), device='cpu', dtype=torch.float32)
    add_5 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    getitem_5 = rand_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cpu', dtype=torch.float32)
    rsqrt_2 = rand_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cpu', dtype=torch.float32)
    view_5 = rand_strided((25088, 128), (128, 1), device='cpu', dtype=torch.float32)
    addmm_2 = rand_strided((25088, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_7 = rand_strided((25088, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_3 = rand_strided((25088, 128), (128, 1), device='cpu', dtype=torch.float32)
    add_9 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cpu', dtype=torch.float32)
    rsqrt_3 = rand_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cpu', dtype=torch.float32)
    view_10 = rand_strided((25088, 128), (128, 1), device='cpu', dtype=torch.float32)
    addmm_4 = rand_strided((25088, 512), (512, 1), device='cpu', dtype=torch.float32)
    view_12 = rand_strided((25088, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_5 = rand_strided((25088, 128), (128, 1), device='cpu', dtype=torch.float32)
    mul_20 = rand_strided((8, 56, 56, 128), (401408, 1, 7168, 56), device='cpu', dtype=torch.float32)
    permute_15 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    getitem_11 = rand_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cpu', dtype=torch.float32)
    rsqrt_5 = rand_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cpu', dtype=torch.float32)
    view_15 = rand_strided((6272, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_6 = rand_strided((6272, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_17 = rand_strided((6272, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_7 = rand_strided((6272, 256), (256, 1), device='cpu', dtype=torch.float32)
    add_19 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    getitem_13 = rand_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cpu', dtype=torch.float32)
    rsqrt_6 = rand_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cpu', dtype=torch.float32)
    view_20 = rand_strided((6272, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_8 = rand_strided((6272, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_22 = rand_strided((6272, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_9 = rand_strided((6272, 256), (256, 1), device='cpu', dtype=torch.float32)
    add_23 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    getitem_15 = rand_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cpu', dtype=torch.float32)
    rsqrt_7 = rand_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cpu', dtype=torch.float32)
    view_25 = rand_strided((6272, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((6272, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_27 = rand_strided((6272, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_11 = rand_strided((6272, 256), (256, 1), device='cpu', dtype=torch.float32)
    mul_40 = rand_strided((8, 28, 28, 256), (200704, 1, 7168, 28), device='cpu', dtype=torch.float32)
    permute_29 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_19 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_9 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_30 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_12 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_32 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_13 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_33 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_21 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_10 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_35 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_14 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_37 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_15 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_37 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_23 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_11 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_40 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_16 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_42 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_17 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_41 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_25 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_12 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_45 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_18 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_47 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_19 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_45 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_27 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_13 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_50 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_20 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_52 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_21 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_49 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_29 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_14 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_55 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_57 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_23 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_53 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_31 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_15 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_60 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_24 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_62 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_25 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_57 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_33 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_16 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_65 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_26 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_67 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_27 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_61 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_35 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_17 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_70 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_28 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_72 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_29 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_65 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_37 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_18 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_75 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_30 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_77 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_31 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_69 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_39 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_19 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_80 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_32 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_82 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_33 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_73 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_41 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_20 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_85 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_87 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_35 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_77 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_43 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_21 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_90 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_36 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_92 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_37 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_81 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_45 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_22 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_95 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_38 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_97 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_39 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_85 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_47 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_23 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_100 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_40 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_102 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_41 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_89 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_49 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_24 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_105 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_42 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_107 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_43 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_93 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_51 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_25 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_110 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_44 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_112 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_45 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_97 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_53 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_26 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_115 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_46 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_117 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_47 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_101 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_55 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_27 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_120 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_48 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_122 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_49 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_105 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_57 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_28 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_125 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_50 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_127 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_51 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_109 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_59 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_29 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_130 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_52 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_132 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_53 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_113 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_61 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_30 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_135 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_54 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_137 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_55 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_117 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_63 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_31 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_140 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_56 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_142 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_57 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_121 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_65 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_32 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_145 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_58 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_147 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_59 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_125 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_67 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_33 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_150 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_60 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_152 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_61 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_129 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_69 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_34 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_155 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_62 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_157 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_63 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    add_133 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    getitem_71 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    rsqrt_35 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    view_160 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    addmm_64 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    view_162 = rand_strided((1568, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    addmm_65 = rand_strided((1568, 512), (512, 1), device='cpu', dtype=torch.float32)
    mul_204 = rand_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cpu', dtype=torch.float32)
    permute_139 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    getitem_75 = rand_strided((8, 7, 7, 1), (49, 1, 7, 7), device='cpu', dtype=torch.float32)
    rsqrt_37 = rand_strided((8, 7, 7, 1), (49, 1, 7, 7), device='cpu', dtype=torch.float32)
    view_165 = rand_strided((392, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_66 = rand_strided((392, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_167 = rand_strided((392, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    addmm_67 = rand_strided((392, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    add_143 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    getitem_77 = rand_strided((8, 7, 7, 1), (49, 1, 7, 7), device='cpu', dtype=torch.float32)
    rsqrt_38 = rand_strided((8, 7, 7, 1), (49, 1, 7, 7), device='cpu', dtype=torch.float32)
    view_170 = rand_strided((392, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_68 = rand_strided((392, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_172 = rand_strided((392, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    addmm_69 = rand_strided((392, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    add_147 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    getitem_79 = rand_strided((8, 7, 7, 1), (49, 1, 7, 7), device='cpu', dtype=torch.float32)
    rsqrt_39 = rand_strided((8, 7, 7, 1), (49, 1, 7, 7), device='cpu', dtype=torch.float32)
    view_175 = rand_strided((392, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_70 = rand_strided((392, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_177 = rand_strided((392, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    addmm_71 = rand_strided((392, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    mul_224 = rand_strided((8, 1, 1, 1024), (1024, 1, 1024, 1), device='cpu', dtype=torch.float32)
    clone_73 = rand_strided((8, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_155 = rand_strided((1000, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    permute_162 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_166 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_172 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_176 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_182 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_186 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_5 = rand_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    permute_194 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_198 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_204 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_208 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_214 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_218 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_224 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_228 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_234 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_238 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_244 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_248 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_254 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_258 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_264 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_268 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_274 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_278 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_284 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_288 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_294 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_298 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_304 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_308 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_314 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_318 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_324 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_328 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_334 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_338 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_344 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_348 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_354 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_358 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_364 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_368 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_374 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_378 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_384 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_388 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_394 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_398 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_404 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_408 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_414 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_418 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_424 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_428 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_434 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_438 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_444 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_448 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_454 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    permute_458 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    div_33 = rand_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cpu', dtype=torch.float32)
    permute_466 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_470 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_476 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_480 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_486 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_490 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_37 = rand_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cpu', dtype=torch.float32)
    permute_498 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_502 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_508 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_512 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_518 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    permute_522 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    div_41 = rand_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_110, primals_111, primals_113, primals_114, primals_116, primals_117, primals_119, primals_121, primals_127, primals_133, primals_139, primals_141, primals_147, primals_153, primals_159, primals_161, primals_167, primals_173, primals_179, primals_185, primals_191, primals_197, primals_203, primals_209, primals_215, primals_221, primals_227, primals_233, primals_239, primals_245, primals_251, primals_257, primals_263, primals_269, primals_275, primals_281, primals_287, primals_293, primals_299, primals_305, primals_311, primals_317, primals_323, primals_325, primals_331, primals_337, primals_345, mul, permute_1, convolution_1, getitem_3, rsqrt_1, view, addmm, view_2, addmm_1, add_5, convolution_2, getitem_5, rsqrt_2, view_5, addmm_2, view_7, addmm_3, add_9, convolution_3, getitem_7, rsqrt_3, view_10, addmm_4, view_12, addmm_5, mul_20, permute_15, convolution_4, convolution_5, getitem_11, rsqrt_5, view_15, addmm_6, view_17, addmm_7, add_19, convolution_6, getitem_13, rsqrt_6, view_20, addmm_8, view_22, addmm_9, add_23, convolution_7, getitem_15, rsqrt_7, view_25, addmm_10, view_27, addmm_11, mul_40, permute_29, convolution_8, convolution_9, getitem_19, rsqrt_9, view_30, addmm_12, view_32, addmm_13, add_33, convolution_10, getitem_21, rsqrt_10, view_35, addmm_14, view_37, addmm_15, add_37, convolution_11, getitem_23, rsqrt_11, view_40, addmm_16, view_42, addmm_17, add_41, convolution_12, getitem_25, rsqrt_12, view_45, addmm_18, view_47, addmm_19, add_45, convolution_13, getitem_27, rsqrt_13, view_50, addmm_20, view_52, addmm_21, add_49, convolution_14, getitem_29, rsqrt_14, view_55, addmm_22, view_57, addmm_23, add_53, convolution_15, getitem_31, rsqrt_15, view_60, addmm_24, view_62, addmm_25, add_57, convolution_16, getitem_33, rsqrt_16, view_65, addmm_26, view_67, addmm_27, add_61, convolution_17, getitem_35, rsqrt_17, view_70, addmm_28, view_72, addmm_29, add_65, convolution_18, getitem_37, rsqrt_18, view_75, addmm_30, view_77, addmm_31, add_69, convolution_19, getitem_39, rsqrt_19, view_80, addmm_32, view_82, addmm_33, add_73, convolution_20, getitem_41, rsqrt_20, view_85, addmm_34, view_87, addmm_35, add_77, convolution_21, getitem_43, rsqrt_21, view_90, addmm_36, view_92, addmm_37, add_81, convolution_22, getitem_45, rsqrt_22, view_95, addmm_38, view_97, addmm_39, add_85, convolution_23, getitem_47, rsqrt_23, view_100, addmm_40, view_102, addmm_41, add_89, convolution_24, getitem_49, rsqrt_24, view_105, addmm_42, view_107, addmm_43, add_93, convolution_25, getitem_51, rsqrt_25, view_110, addmm_44, view_112, addmm_45, add_97, convolution_26, getitem_53, rsqrt_26, view_115, addmm_46, view_117, addmm_47, add_101, convolution_27, getitem_55, rsqrt_27, view_120, addmm_48, view_122, addmm_49, add_105, convolution_28, getitem_57, rsqrt_28, view_125, addmm_50, view_127, addmm_51, add_109, convolution_29, getitem_59, rsqrt_29, view_130, addmm_52, view_132, addmm_53, add_113, convolution_30, getitem_61, rsqrt_30, view_135, addmm_54, view_137, addmm_55, add_117, convolution_31, getitem_63, rsqrt_31, view_140, addmm_56, view_142, addmm_57, add_121, convolution_32, getitem_65, rsqrt_32, view_145, addmm_58, view_147, addmm_59, add_125, convolution_33, getitem_67, rsqrt_33, view_150, addmm_60, view_152, addmm_61, add_129, convolution_34, getitem_69, rsqrt_34, view_155, addmm_62, view_157, addmm_63, add_133, convolution_35, getitem_71, rsqrt_35, view_160, addmm_64, view_162, addmm_65, mul_204, permute_139, convolution_36, convolution_37, getitem_75, rsqrt_37, view_165, addmm_66, view_167, addmm_67, add_143, convolution_38, getitem_77, rsqrt_38, view_170, addmm_68, view_172, addmm_69, add_147, convolution_39, getitem_79, rsqrt_39, view_175, addmm_70, view_177, addmm_71, mul_224, clone_73, permute_155, div, permute_162, permute_166, permute_172, permute_176, permute_182, permute_186, div_5, permute_194, permute_198, permute_204, permute_208, permute_214, permute_218, permute_224, permute_228, permute_234, permute_238, permute_244, permute_248, permute_254, permute_258, permute_264, permute_268, permute_274, permute_278, permute_284, permute_288, permute_294, permute_298, permute_304, permute_308, permute_314, permute_318, permute_324, permute_328, permute_334, permute_338, permute_344, permute_348, permute_354, permute_358, permute_364, permute_368, permute_374, permute_378, permute_384, permute_388, permute_394, permute_398, permute_404, permute_408, permute_414, permute_418, permute_424, permute_428, permute_434, permute_438, permute_444, permute_448, permute_454, permute_458, div_33, permute_466, permute_470, permute_476, permute_480, permute_486, permute_490, div_37, permute_498, permute_502, permute_508, permute_512, permute_518, permute_522, div_41, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('convnext_base', benchmark_compiled_module)
