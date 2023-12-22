
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


cpp_fused_native_dropout_backward_native_layer_norm_backward_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp11 = out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = static_cast<float>(768.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 - tmp13;
                    auto tmp15 = at::vec::Vectorized<float>(tmp0);
                    auto tmp16 = tmp15 * tmp14;
                    tmp16.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr2[static_cast<long>(x0)];
                auto tmp1 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr5[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_clone_sum_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_arange_clone_index_add_new_zeros_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(long* out_ptr0,
                       float* out_ptr1,
                       long* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1179648L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = c10::convert<long>(x0);
                out_ptr0[static_cast<long>(x0)] = tmp0;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1179648L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49152L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x2 + (16384L*x1) + (98304L*x0))];
                        out_ptr2[static_cast<long>(x2 + (49152L*x1) + (196608L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14)
{
    auto out_ptr8 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0)];
                            auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = static_cast<long>(770);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = [&]
                            {
                                auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                                auto tmp6 = static_cast<long>(196864);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                    auto tmp10 = static_cast<long>(768);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                            auto tmp22 = c10::convert<float>(tmp21);
                            auto tmp23 = static_cast<float>(1.1111111111111112);
                            auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                            auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                            auto tmp26 = static_cast<float>(0.0);
                            auto tmp27 = tmp0 ? tmp26 : tmp25;
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            tmp_acc0 = tmp_acc0 + tmp29;
                        }
                        out_ptr0[static_cast<long>(x1 + (12L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp30 = out_ptr0[static_cast<long>(x1 + (12L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = static_cast<long>(770);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                            auto tmp6 = static_cast<long>(196864);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                auto tmp10 = static_cast<long>(768);
                                auto tmp11 = tmp9 < tmp10;
                                auto tmp12 = [&]
                                {
                                    auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                    return tmp13;
                                }
                                ;
                                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                auto tmp15 = static_cast<float>(0.0);
                                auto tmp16 = tmp11 ? tmp14 : tmp15;
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp18 = static_cast<float>(0.0);
                            auto tmp19 = tmp7 ? tmp17 : tmp18;
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp22 = c10::convert<float>(tmp21);
                        auto tmp23 = static_cast<float>(1.1111111111111112);
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = static_cast<float>(0.0);
                        auto tmp27 = tmp0 ? tmp26 : tmp25;
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                        auto tmp32 = decltype(tmp29)(tmp29 - tmp31);
                        out_ptr1[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp32;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr5 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr5[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr4[static_cast<long>((-197632L) + x2 + (257L*x1))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr4[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp11 = static_cast<float>(0.0);
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = static_cast<float>(0.0);
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp17 = static_cast<float>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        out_ptr6[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp18;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr7 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                        tmp0.store(out_ptr7 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr6[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        out_ptr7[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(out_ptr8 + static_cast<long>(x0));
                tmp2.store(out_ptr9 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr9 + static_cast<long>(x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp1 = c10::convert<long>(x1);
                        auto tmp2 = static_cast<long>(256);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2);
                            auto tmp6 = static_cast<long>(257);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = in_ptr5[static_cast<long>(x2 + (257L*x1))];
                                auto tmp10 = c10::convert<bool>(tmp9);
                                auto tmp11 = out_ptr8[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp12 = static_cast<float>(0.0);
                                auto tmp13 = tmp10 ? tmp12 : tmp11;
                                return tmp13;
                            }
                            ;
                            auto tmp14 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp15 = static_cast<float>(0.0);
                            auto tmp16 = tmp7 ? tmp14 : tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp18 = static_cast<float>(0.0);
                        auto tmp19 = tmp3 ? tmp17 : tmp18;
                        auto tmp20 = decltype(tmp0)(tmp0 + tmp19);
                        in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp20;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr10 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(255L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(248L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr10 + static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(248L); x2<static_cast<long>(255L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr10[static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr10 + static_cast<long>(x0));
                tmp0.store(out_ptr11 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr11 + static_cast<long>(131328L + x2 + (513L*x1) + (525312L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr11 + static_cast<long>(x0));
                tmp0.store(out_ptr12 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr12 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr12[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x2);
                            auto tmp4 = static_cast<long>(255);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(x3);
                                auto tmp8 = static_cast<long>(258);
                                auto tmp9 = tmp7 >= tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_out_ptr0[static_cast<long>(256L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                auto tmp13 = static_cast<float>(0.0);
                                auto tmp14 = tmp9 ? tmp12 : tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp16 = static_cast<float>(0.0);
                            auto tmp17 = tmp5 ? tmp15 : tmp16;
                            auto tmp18 = tmp2 ? tmp17 : tmp16;
                            auto tmp19 = tmp3 >= tmp4;
                            auto tmp20 = static_cast<long>(511);
                            auto tmp21 = tmp3 < tmp20;
                            auto tmp22 = tmp19 & tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(x3);
                                auto tmp25 = static_cast<long>(257);
                                auto tmp26 = tmp24 >= tmp25;
                                auto tmp27 = [&]
                                {
                                    auto tmp28 = out_ptr10[static_cast<long>(256L + x3 + (513L*x2) + (131328L*x1) + (525312L*x0))];
                                    return tmp28;
                                }
                                ;
                                auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                                auto tmp30 = static_cast<float>(0.0);
                                auto tmp31 = tmp26 ? tmp29 : tmp30;
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp33 = tmp22 ? tmp32 : tmp16;
                            auto tmp34 = decltype(tmp18)(tmp18 + tmp33);
                            auto tmp35 = static_cast<int>(2);
                            auto tmp36 = tmp0 == tmp35;
                            auto tmp37 = static_cast<long>(256);
                            auto tmp38 = tmp3 >= tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x3);
                                auto tmp41 = static_cast<long>(257);
                                auto tmp42 = tmp40 < tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = out_ptr11[static_cast<long>(262912L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp44;
                                }
                                ;
                                auto tmp45 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp46 = static_cast<float>(0.0);
                                auto tmp47 = tmp42 ? tmp45 : tmp46;
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp38 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp49 = tmp38 ? tmp48 : tmp16;
                            auto tmp50 = tmp36 ? tmp49 : tmp16;
                            auto tmp51 = decltype(tmp34)(tmp34 + tmp50);
                            out_ptr13[static_cast<long>(x3 + (513L*x2) + (262656L*x1) + (787968L*x0))] = tmp51;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = static_cast<long>(513);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = out_ptr13[static_cast<long>(x3 + (512L*x2) + (262656L*x1) + (262656L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (787968L*x0) + (787968L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                auto tmp5 = c10::convert<long>(c10::div_floor_integer((x3 + (512L*x2)), 513L));
                                auto tmp6 = static_cast<long>(256);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x3 + (512L*x2))) % static_cast<long>(513L));
                                    auto tmp10 = static_cast<long>(257);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = out_ptr12[static_cast<long>(256L + x3 + (512L*x2) + (131328L*x1) + (131328L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (525312L*x0) + (525312L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                auto tmp20 = decltype(tmp4)(tmp4 + tmp19);
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                            out_ptr14[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (786432L*x0))] = tmp21;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_arange_clone_index_add_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       long* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       long* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = c10::convert<long>(x0);
                out_ptr0[static_cast<long>(x0)] = tmp0;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr0[static_cast<long>(x3 + (64L*x0) + (768L*x2) + (196608L*x1))];
                            out_ptr3[static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (98304L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_as_strided_scatter_div_view_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp3.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = c10::convert<int>(256L + x0);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<int>(1536);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = tmp2 & tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(16384L + (64L*x0) + (98304L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp5)); })();
                        return tmp7;
                    }
                    ;
                    auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_11 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_clone_sum_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_new_zeros_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1179648L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14)
{
    auto out_ptr8 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0)];
                            auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = static_cast<long>(770);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = [&]
                            {
                                auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                                auto tmp6 = static_cast<long>(196864);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                    auto tmp10 = static_cast<long>(768);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                            auto tmp22 = c10::convert<float>(tmp21);
                            auto tmp23 = static_cast<float>(1.1111111111111112);
                            auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                            auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                            auto tmp26 = static_cast<float>(0.0);
                            auto tmp27 = tmp0 ? tmp26 : tmp25;
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            tmp_acc0 = tmp_acc0 + tmp29;
                        }
                        out_ptr0[static_cast<long>(x1 + (12L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp30 = out_ptr0[static_cast<long>(x1 + (12L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = static_cast<long>(770);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                            auto tmp6 = static_cast<long>(196864);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                auto tmp10 = static_cast<long>(768);
                                auto tmp11 = tmp9 < tmp10;
                                auto tmp12 = [&]
                                {
                                    auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                    return tmp13;
                                }
                                ;
                                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                auto tmp15 = static_cast<float>(0.0);
                                auto tmp16 = tmp11 ? tmp14 : tmp15;
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp18 = static_cast<float>(0.0);
                            auto tmp19 = tmp7 ? tmp17 : tmp18;
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp22 = c10::convert<float>(tmp21);
                        auto tmp23 = static_cast<float>(1.1111111111111112);
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = static_cast<float>(0.0);
                        auto tmp27 = tmp0 ? tmp26 : tmp25;
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                        auto tmp32 = decltype(tmp29)(tmp29 - tmp31);
                        out_ptr1[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp32;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr5 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr5[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr4[static_cast<long>((-197632L) + x2 + (257L*x1))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr4[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp11 = static_cast<float>(0.0);
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = static_cast<float>(0.0);
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp17 = static_cast<float>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        out_ptr6[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp18;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr7 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                        tmp0.store(out_ptr7 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr6[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        out_ptr7[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(out_ptr8 + static_cast<long>(x0));
                tmp2.store(out_ptr9 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr9 + static_cast<long>(x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp1 = c10::convert<long>(x1);
                        auto tmp2 = static_cast<long>(256);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2);
                            auto tmp6 = static_cast<long>(257);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = in_ptr5[static_cast<long>(x2 + (257L*x1))];
                                auto tmp10 = c10::convert<bool>(tmp9);
                                auto tmp11 = out_ptr8[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp12 = static_cast<float>(0.0);
                                auto tmp13 = tmp10 ? tmp12 : tmp11;
                                return tmp13;
                            }
                            ;
                            auto tmp14 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp15 = static_cast<float>(0.0);
                            auto tmp16 = tmp7 ? tmp14 : tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp18 = static_cast<float>(0.0);
                        auto tmp19 = tmp3 ? tmp17 : tmp18;
                        auto tmp20 = decltype(tmp0)(tmp0 + tmp19);
                        in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp20;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr10 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(255L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(248L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr10 + static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(248L); x2<static_cast<long>(255L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr10[static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr10 + static_cast<long>(x0));
                tmp0.store(out_ptr11 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr11 + static_cast<long>(131328L + x2 + (513L*x1) + (525312L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr11 + static_cast<long>(x0));
                tmp0.store(out_ptr12 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr12 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr12[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x2);
                            auto tmp4 = static_cast<long>(255);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(x3);
                                auto tmp8 = static_cast<long>(258);
                                auto tmp9 = tmp7 >= tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_out_ptr0[static_cast<long>(256L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                auto tmp13 = static_cast<float>(0.0);
                                auto tmp14 = tmp9 ? tmp12 : tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp16 = static_cast<float>(0.0);
                            auto tmp17 = tmp5 ? tmp15 : tmp16;
                            auto tmp18 = tmp2 ? tmp17 : tmp16;
                            auto tmp19 = tmp3 >= tmp4;
                            auto tmp20 = static_cast<long>(511);
                            auto tmp21 = tmp3 < tmp20;
                            auto tmp22 = tmp19 & tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(x3);
                                auto tmp25 = static_cast<long>(257);
                                auto tmp26 = tmp24 >= tmp25;
                                auto tmp27 = [&]
                                {
                                    auto tmp28 = out_ptr10[static_cast<long>(256L + x3 + (513L*x2) + (131328L*x1) + (525312L*x0))];
                                    return tmp28;
                                }
                                ;
                                auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                                auto tmp30 = static_cast<float>(0.0);
                                auto tmp31 = tmp26 ? tmp29 : tmp30;
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp33 = tmp22 ? tmp32 : tmp16;
                            auto tmp34 = decltype(tmp18)(tmp18 + tmp33);
                            auto tmp35 = static_cast<int>(2);
                            auto tmp36 = tmp0 == tmp35;
                            auto tmp37 = static_cast<long>(256);
                            auto tmp38 = tmp3 >= tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x3);
                                auto tmp41 = static_cast<long>(257);
                                auto tmp42 = tmp40 < tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = out_ptr11[static_cast<long>(262912L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp44;
                                }
                                ;
                                auto tmp45 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp46 = static_cast<float>(0.0);
                                auto tmp47 = tmp42 ? tmp45 : tmp46;
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp38 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp49 = tmp38 ? tmp48 : tmp16;
                            auto tmp50 = tmp36 ? tmp49 : tmp16;
                            auto tmp51 = decltype(tmp34)(tmp34 + tmp50);
                            out_ptr13[static_cast<long>(x3 + (513L*x2) + (262656L*x1) + (787968L*x0))] = tmp51;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = static_cast<long>(513);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = out_ptr13[static_cast<long>(x3 + (512L*x2) + (262656L*x1) + (262656L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (787968L*x0) + (787968L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                auto tmp5 = c10::convert<long>(c10::div_floor_integer((x3 + (512L*x2)), 513L));
                                auto tmp6 = static_cast<long>(256);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x3 + (512L*x2))) % static_cast<long>(513L));
                                    auto tmp10 = static_cast<long>(257);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = out_ptr12[static_cast<long>(256L + x3 + (512L*x2) + (131328L*x1) + (131328L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (525312L*x0) + (525312L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                auto tmp20 = decltype(tmp4)(tmp4 + tmp19);
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                            out_ptr14[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (786432L*x0))] = tmp21;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_index_add_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_as_strided_scatter_div_view_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp3.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = c10::convert<int>(256L + x0);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<int>(1536);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = tmp2 & tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(16384L + (64L*x0) + (98304L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp5)); })();
                        return tmp7;
                    }
                    ;
                    auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_22 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_clone_sum_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_new_zeros_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1179648L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14)
{
    auto out_ptr8 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0)];
                            auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = static_cast<long>(770);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = [&]
                            {
                                auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                                auto tmp6 = static_cast<long>(196864);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                    auto tmp10 = static_cast<long>(768);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                            auto tmp22 = c10::convert<float>(tmp21);
                            auto tmp23 = static_cast<float>(1.1111111111111112);
                            auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                            auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                            auto tmp26 = static_cast<float>(0.0);
                            auto tmp27 = tmp0 ? tmp26 : tmp25;
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            tmp_acc0 = tmp_acc0 + tmp29;
                        }
                        out_ptr0[static_cast<long>(x1 + (12L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp30 = out_ptr0[static_cast<long>(x1 + (12L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = static_cast<long>(770);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                            auto tmp6 = static_cast<long>(196864);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                auto tmp10 = static_cast<long>(768);
                                auto tmp11 = tmp9 < tmp10;
                                auto tmp12 = [&]
                                {
                                    auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                    return tmp13;
                                }
                                ;
                                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                auto tmp15 = static_cast<float>(0.0);
                                auto tmp16 = tmp11 ? tmp14 : tmp15;
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp18 = static_cast<float>(0.0);
                            auto tmp19 = tmp7 ? tmp17 : tmp18;
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp22 = c10::convert<float>(tmp21);
                        auto tmp23 = static_cast<float>(1.1111111111111112);
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = static_cast<float>(0.0);
                        auto tmp27 = tmp0 ? tmp26 : tmp25;
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                        auto tmp32 = decltype(tmp29)(tmp29 - tmp31);
                        out_ptr1[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp32;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr5 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr5[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr4[static_cast<long>((-197632L) + x2 + (257L*x1))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr4[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp11 = static_cast<float>(0.0);
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = static_cast<float>(0.0);
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp17 = static_cast<float>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        out_ptr6[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp18;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr7 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                        tmp0.store(out_ptr7 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr6[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        out_ptr7[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(out_ptr8 + static_cast<long>(x0));
                tmp2.store(out_ptr9 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr9 + static_cast<long>(x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp1 = c10::convert<long>(x1);
                        auto tmp2 = static_cast<long>(256);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2);
                            auto tmp6 = static_cast<long>(257);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = in_ptr5[static_cast<long>(x2 + (257L*x1))];
                                auto tmp10 = c10::convert<bool>(tmp9);
                                auto tmp11 = out_ptr8[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp12 = static_cast<float>(0.0);
                                auto tmp13 = tmp10 ? tmp12 : tmp11;
                                return tmp13;
                            }
                            ;
                            auto tmp14 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp15 = static_cast<float>(0.0);
                            auto tmp16 = tmp7 ? tmp14 : tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp18 = static_cast<float>(0.0);
                        auto tmp19 = tmp3 ? tmp17 : tmp18;
                        auto tmp20 = decltype(tmp0)(tmp0 + tmp19);
                        in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp20;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr10 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(255L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(248L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr10 + static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(248L); x2<static_cast<long>(255L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr10[static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr10 + static_cast<long>(x0));
                tmp0.store(out_ptr11 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr11 + static_cast<long>(131328L + x2 + (513L*x1) + (525312L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr11 + static_cast<long>(x0));
                tmp0.store(out_ptr12 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr12 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr12[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x2);
                            auto tmp4 = static_cast<long>(255);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(x3);
                                auto tmp8 = static_cast<long>(258);
                                auto tmp9 = tmp7 >= tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_out_ptr0[static_cast<long>(256L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                auto tmp13 = static_cast<float>(0.0);
                                auto tmp14 = tmp9 ? tmp12 : tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp16 = static_cast<float>(0.0);
                            auto tmp17 = tmp5 ? tmp15 : tmp16;
                            auto tmp18 = tmp2 ? tmp17 : tmp16;
                            auto tmp19 = tmp3 >= tmp4;
                            auto tmp20 = static_cast<long>(511);
                            auto tmp21 = tmp3 < tmp20;
                            auto tmp22 = tmp19 & tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(x3);
                                auto tmp25 = static_cast<long>(257);
                                auto tmp26 = tmp24 >= tmp25;
                                auto tmp27 = [&]
                                {
                                    auto tmp28 = out_ptr10[static_cast<long>(256L + x3 + (513L*x2) + (131328L*x1) + (525312L*x0))];
                                    return tmp28;
                                }
                                ;
                                auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                                auto tmp30 = static_cast<float>(0.0);
                                auto tmp31 = tmp26 ? tmp29 : tmp30;
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp33 = tmp22 ? tmp32 : tmp16;
                            auto tmp34 = decltype(tmp18)(tmp18 + tmp33);
                            auto tmp35 = static_cast<int>(2);
                            auto tmp36 = tmp0 == tmp35;
                            auto tmp37 = static_cast<long>(256);
                            auto tmp38 = tmp3 >= tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x3);
                                auto tmp41 = static_cast<long>(257);
                                auto tmp42 = tmp40 < tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = out_ptr11[static_cast<long>(262912L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp44;
                                }
                                ;
                                auto tmp45 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp46 = static_cast<float>(0.0);
                                auto tmp47 = tmp42 ? tmp45 : tmp46;
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp38 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp49 = tmp38 ? tmp48 : tmp16;
                            auto tmp50 = tmp36 ? tmp49 : tmp16;
                            auto tmp51 = decltype(tmp34)(tmp34 + tmp50);
                            out_ptr13[static_cast<long>(x3 + (513L*x2) + (262656L*x1) + (787968L*x0))] = tmp51;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = static_cast<long>(513);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = out_ptr13[static_cast<long>(x3 + (512L*x2) + (262656L*x1) + (262656L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (787968L*x0) + (787968L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                auto tmp5 = c10::convert<long>(c10::div_floor_integer((x3 + (512L*x2)), 513L));
                                auto tmp6 = static_cast<long>(256);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x3 + (512L*x2))) % static_cast<long>(513L));
                                    auto tmp10 = static_cast<long>(257);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = out_ptr12[static_cast<long>(256L + x3 + (512L*x2) + (131328L*x1) + (131328L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (525312L*x0) + (525312L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                auto tmp20 = decltype(tmp4)(tmp4 + tmp19);
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                            out_ptr14[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (786432L*x0))] = tmp21;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_index_add_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_as_strided_scatter_div_view_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp3.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = c10::convert<int>(256L + x0);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<int>(1536);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = tmp2 & tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(16384L + (64L*x0) + (98304L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp5)); })();
                        return tmp7;
                    }
                    ;
                    auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_33 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_clone_sum_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_new_zeros_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1179648L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14)
{
    auto out_ptr8 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0)];
                            auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = static_cast<long>(770);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = [&]
                            {
                                auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                                auto tmp6 = static_cast<long>(196864);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                    auto tmp10 = static_cast<long>(768);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                            auto tmp22 = c10::convert<float>(tmp21);
                            auto tmp23 = static_cast<float>(1.1111111111111112);
                            auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                            auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                            auto tmp26 = static_cast<float>(0.0);
                            auto tmp27 = tmp0 ? tmp26 : tmp25;
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            tmp_acc0 = tmp_acc0 + tmp29;
                        }
                        out_ptr0[static_cast<long>(x1 + (12L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp30 = out_ptr0[static_cast<long>(x1 + (12L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = static_cast<long>(770);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                            auto tmp6 = static_cast<long>(196864);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                auto tmp10 = static_cast<long>(768);
                                auto tmp11 = tmp9 < tmp10;
                                auto tmp12 = [&]
                                {
                                    auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                    return tmp13;
                                }
                                ;
                                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                auto tmp15 = static_cast<float>(0.0);
                                auto tmp16 = tmp11 ? tmp14 : tmp15;
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp18 = static_cast<float>(0.0);
                            auto tmp19 = tmp7 ? tmp17 : tmp18;
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp22 = c10::convert<float>(tmp21);
                        auto tmp23 = static_cast<float>(1.1111111111111112);
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = static_cast<float>(0.0);
                        auto tmp27 = tmp0 ? tmp26 : tmp25;
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                        auto tmp32 = decltype(tmp29)(tmp29 - tmp31);
                        out_ptr1[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp32;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr5 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr5[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr4[static_cast<long>((-197632L) + x2 + (257L*x1))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr4[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp11 = static_cast<float>(0.0);
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = static_cast<float>(0.0);
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp17 = static_cast<float>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        out_ptr6[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp18;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr7 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                        tmp0.store(out_ptr7 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr6[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        out_ptr7[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(out_ptr8 + static_cast<long>(x0));
                tmp2.store(out_ptr9 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr9 + static_cast<long>(x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp1 = c10::convert<long>(x1);
                        auto tmp2 = static_cast<long>(256);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2);
                            auto tmp6 = static_cast<long>(257);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = in_ptr5[static_cast<long>(x2 + (257L*x1))];
                                auto tmp10 = c10::convert<bool>(tmp9);
                                auto tmp11 = out_ptr8[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp12 = static_cast<float>(0.0);
                                auto tmp13 = tmp10 ? tmp12 : tmp11;
                                return tmp13;
                            }
                            ;
                            auto tmp14 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp15 = static_cast<float>(0.0);
                            auto tmp16 = tmp7 ? tmp14 : tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp18 = static_cast<float>(0.0);
                        auto tmp19 = tmp3 ? tmp17 : tmp18;
                        auto tmp20 = decltype(tmp0)(tmp0 + tmp19);
                        in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp20;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr10 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(255L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(248L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr10 + static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(248L); x2<static_cast<long>(255L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr10[static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr10 + static_cast<long>(x0));
                tmp0.store(out_ptr11 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr11 + static_cast<long>(131328L + x2 + (513L*x1) + (525312L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr11 + static_cast<long>(x0));
                tmp0.store(out_ptr12 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr12 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr12[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x2);
                            auto tmp4 = static_cast<long>(255);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(x3);
                                auto tmp8 = static_cast<long>(258);
                                auto tmp9 = tmp7 >= tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_out_ptr0[static_cast<long>(256L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                auto tmp13 = static_cast<float>(0.0);
                                auto tmp14 = tmp9 ? tmp12 : tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp16 = static_cast<float>(0.0);
                            auto tmp17 = tmp5 ? tmp15 : tmp16;
                            auto tmp18 = tmp2 ? tmp17 : tmp16;
                            auto tmp19 = tmp3 >= tmp4;
                            auto tmp20 = static_cast<long>(511);
                            auto tmp21 = tmp3 < tmp20;
                            auto tmp22 = tmp19 & tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(x3);
                                auto tmp25 = static_cast<long>(257);
                                auto tmp26 = tmp24 >= tmp25;
                                auto tmp27 = [&]
                                {
                                    auto tmp28 = out_ptr10[static_cast<long>(256L + x3 + (513L*x2) + (131328L*x1) + (525312L*x0))];
                                    return tmp28;
                                }
                                ;
                                auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                                auto tmp30 = static_cast<float>(0.0);
                                auto tmp31 = tmp26 ? tmp29 : tmp30;
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp33 = tmp22 ? tmp32 : tmp16;
                            auto tmp34 = decltype(tmp18)(tmp18 + tmp33);
                            auto tmp35 = static_cast<int>(2);
                            auto tmp36 = tmp0 == tmp35;
                            auto tmp37 = static_cast<long>(256);
                            auto tmp38 = tmp3 >= tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x3);
                                auto tmp41 = static_cast<long>(257);
                                auto tmp42 = tmp40 < tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = out_ptr11[static_cast<long>(262912L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp44;
                                }
                                ;
                                auto tmp45 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp46 = static_cast<float>(0.0);
                                auto tmp47 = tmp42 ? tmp45 : tmp46;
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp38 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp49 = tmp38 ? tmp48 : tmp16;
                            auto tmp50 = tmp36 ? tmp49 : tmp16;
                            auto tmp51 = decltype(tmp34)(tmp34 + tmp50);
                            out_ptr13[static_cast<long>(x3 + (513L*x2) + (262656L*x1) + (787968L*x0))] = tmp51;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = static_cast<long>(513);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = out_ptr13[static_cast<long>(x3 + (512L*x2) + (262656L*x1) + (262656L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (787968L*x0) + (787968L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                auto tmp5 = c10::convert<long>(c10::div_floor_integer((x3 + (512L*x2)), 513L));
                                auto tmp6 = static_cast<long>(256);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x3 + (512L*x2))) % static_cast<long>(513L));
                                    auto tmp10 = static_cast<long>(257);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = out_ptr12[static_cast<long>(256L + x3 + (512L*x2) + (131328L*x1) + (131328L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (525312L*x0) + (525312L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                auto tmp20 = decltype(tmp4)(tmp4 + tmp19);
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                            out_ptr14[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (786432L*x0))] = tmp21;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_index_add_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_as_strided_scatter_div_view_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp3.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = c10::convert<int>(256L + x0);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<int>(1536);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = tmp2 & tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(16384L + (64L*x0) + (98304L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp5)); })();
                        return tmp7;
                    }
                    ;
                    auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_44 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_clone_sum_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_new_zeros_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1179648L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14)
{
    auto out_ptr8 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0)];
                            auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = static_cast<long>(770);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = [&]
                            {
                                auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                                auto tmp6 = static_cast<long>(196864);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                    auto tmp10 = static_cast<long>(768);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                            auto tmp22 = c10::convert<float>(tmp21);
                            auto tmp23 = static_cast<float>(1.1111111111111112);
                            auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                            auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                            auto tmp26 = static_cast<float>(0.0);
                            auto tmp27 = tmp0 ? tmp26 : tmp25;
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            tmp_acc0 = tmp_acc0 + tmp29;
                        }
                        out_ptr0[static_cast<long>(x1 + (12L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp30 = out_ptr0[static_cast<long>(x1 + (12L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = static_cast<long>(770);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                            auto tmp6 = static_cast<long>(196864);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                auto tmp10 = static_cast<long>(768);
                                auto tmp11 = tmp9 < tmp10;
                                auto tmp12 = [&]
                                {
                                    auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                    return tmp13;
                                }
                                ;
                                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                auto tmp15 = static_cast<float>(0.0);
                                auto tmp16 = tmp11 ? tmp14 : tmp15;
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp18 = static_cast<float>(0.0);
                            auto tmp19 = tmp7 ? tmp17 : tmp18;
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp22 = c10::convert<float>(tmp21);
                        auto tmp23 = static_cast<float>(1.1111111111111112);
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = static_cast<float>(0.0);
                        auto tmp27 = tmp0 ? tmp26 : tmp25;
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                        auto tmp32 = decltype(tmp29)(tmp29 - tmp31);
                        out_ptr1[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp32;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr5 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr5[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr4[static_cast<long>((-197632L) + x2 + (257L*x1))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr4[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp11 = static_cast<float>(0.0);
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = static_cast<float>(0.0);
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp17 = static_cast<float>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        out_ptr6[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp18;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr7 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                        tmp0.store(out_ptr7 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr6[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        out_ptr7[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(out_ptr8 + static_cast<long>(x0));
                tmp2.store(out_ptr9 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr9 + static_cast<long>(x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp1 = c10::convert<long>(x1);
                        auto tmp2 = static_cast<long>(256);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2);
                            auto tmp6 = static_cast<long>(257);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = in_ptr5[static_cast<long>(x2 + (257L*x1))];
                                auto tmp10 = c10::convert<bool>(tmp9);
                                auto tmp11 = out_ptr8[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp12 = static_cast<float>(0.0);
                                auto tmp13 = tmp10 ? tmp12 : tmp11;
                                return tmp13;
                            }
                            ;
                            auto tmp14 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp15 = static_cast<float>(0.0);
                            auto tmp16 = tmp7 ? tmp14 : tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp18 = static_cast<float>(0.0);
                        auto tmp19 = tmp3 ? tmp17 : tmp18;
                        auto tmp20 = decltype(tmp0)(tmp0 + tmp19);
                        in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp20;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr10 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(255L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(248L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr10 + static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(248L); x2<static_cast<long>(255L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr10[static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr10 + static_cast<long>(x0));
                tmp0.store(out_ptr11 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr11 + static_cast<long>(131328L + x2 + (513L*x1) + (525312L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr11 + static_cast<long>(x0));
                tmp0.store(out_ptr12 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr12 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr12[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x2);
                            auto tmp4 = static_cast<long>(255);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(x3);
                                auto tmp8 = static_cast<long>(258);
                                auto tmp9 = tmp7 >= tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_out_ptr0[static_cast<long>(256L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                auto tmp13 = static_cast<float>(0.0);
                                auto tmp14 = tmp9 ? tmp12 : tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp16 = static_cast<float>(0.0);
                            auto tmp17 = tmp5 ? tmp15 : tmp16;
                            auto tmp18 = tmp2 ? tmp17 : tmp16;
                            auto tmp19 = tmp3 >= tmp4;
                            auto tmp20 = static_cast<long>(511);
                            auto tmp21 = tmp3 < tmp20;
                            auto tmp22 = tmp19 & tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(x3);
                                auto tmp25 = static_cast<long>(257);
                                auto tmp26 = tmp24 >= tmp25;
                                auto tmp27 = [&]
                                {
                                    auto tmp28 = out_ptr10[static_cast<long>(256L + x3 + (513L*x2) + (131328L*x1) + (525312L*x0))];
                                    return tmp28;
                                }
                                ;
                                auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                                auto tmp30 = static_cast<float>(0.0);
                                auto tmp31 = tmp26 ? tmp29 : tmp30;
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp33 = tmp22 ? tmp32 : tmp16;
                            auto tmp34 = decltype(tmp18)(tmp18 + tmp33);
                            auto tmp35 = static_cast<int>(2);
                            auto tmp36 = tmp0 == tmp35;
                            auto tmp37 = static_cast<long>(256);
                            auto tmp38 = tmp3 >= tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x3);
                                auto tmp41 = static_cast<long>(257);
                                auto tmp42 = tmp40 < tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = out_ptr11[static_cast<long>(262912L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp44;
                                }
                                ;
                                auto tmp45 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp46 = static_cast<float>(0.0);
                                auto tmp47 = tmp42 ? tmp45 : tmp46;
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp38 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp49 = tmp38 ? tmp48 : tmp16;
                            auto tmp50 = tmp36 ? tmp49 : tmp16;
                            auto tmp51 = decltype(tmp34)(tmp34 + tmp50);
                            out_ptr13[static_cast<long>(x3 + (513L*x2) + (262656L*x1) + (787968L*x0))] = tmp51;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = static_cast<long>(513);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = out_ptr13[static_cast<long>(x3 + (512L*x2) + (262656L*x1) + (262656L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (787968L*x0) + (787968L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                auto tmp5 = c10::convert<long>(c10::div_floor_integer((x3 + (512L*x2)), 513L));
                                auto tmp6 = static_cast<long>(256);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x3 + (512L*x2))) % static_cast<long>(513L));
                                    auto tmp10 = static_cast<long>(257);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = out_ptr12[static_cast<long>(256L + x3 + (512L*x2) + (131328L*x1) + (131328L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (525312L*x0) + (525312L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                auto tmp20 = decltype(tmp4)(tmp4 + tmp19);
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                            out_ptr14[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (786432L*x0))] = tmp21;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_index_add_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_as_strided_scatter_div_view_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp3.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = c10::convert<int>(256L + x0);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<int>(1536);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = tmp2 & tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(16384L + (64L*x0) + (98304L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp5)); })();
                        return tmp7;
                    }
                    ;
                    auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_55 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_clone_sum_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_new_zeros_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1179648L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14)
{
    auto out_ptr8 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0)];
                            auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = static_cast<long>(770);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = [&]
                            {
                                auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                                auto tmp6 = static_cast<long>(196864);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                    auto tmp10 = static_cast<long>(768);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                            auto tmp22 = c10::convert<float>(tmp21);
                            auto tmp23 = static_cast<float>(1.1111111111111112);
                            auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                            auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                            auto tmp26 = static_cast<float>(0.0);
                            auto tmp27 = tmp0 ? tmp26 : tmp25;
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            tmp_acc0 = tmp_acc0 + tmp29;
                        }
                        out_ptr0[static_cast<long>(x1 + (12L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp30 = out_ptr0[static_cast<long>(x1 + (12L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = static_cast<long>(770);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                            auto tmp6 = static_cast<long>(196864);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                auto tmp10 = static_cast<long>(768);
                                auto tmp11 = tmp9 < tmp10;
                                auto tmp12 = [&]
                                {
                                    auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                    return tmp13;
                                }
                                ;
                                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                auto tmp15 = static_cast<float>(0.0);
                                auto tmp16 = tmp11 ? tmp14 : tmp15;
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp18 = static_cast<float>(0.0);
                            auto tmp19 = tmp7 ? tmp17 : tmp18;
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp22 = c10::convert<float>(tmp21);
                        auto tmp23 = static_cast<float>(1.1111111111111112);
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = static_cast<float>(0.0);
                        auto tmp27 = tmp0 ? tmp26 : tmp25;
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                        auto tmp32 = decltype(tmp29)(tmp29 - tmp31);
                        out_ptr1[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp32;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr5 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr5[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr4[static_cast<long>((-197632L) + x2 + (257L*x1))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr4[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp11 = static_cast<float>(0.0);
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = static_cast<float>(0.0);
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp17 = static_cast<float>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        out_ptr6[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp18;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr7 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                        tmp0.store(out_ptr7 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr6[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        out_ptr7[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(out_ptr8 + static_cast<long>(x0));
                tmp2.store(out_ptr9 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr9 + static_cast<long>(x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp1 = c10::convert<long>(x1);
                        auto tmp2 = static_cast<long>(256);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2);
                            auto tmp6 = static_cast<long>(257);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = in_ptr5[static_cast<long>(x2 + (257L*x1))];
                                auto tmp10 = c10::convert<bool>(tmp9);
                                auto tmp11 = out_ptr8[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp12 = static_cast<float>(0.0);
                                auto tmp13 = tmp10 ? tmp12 : tmp11;
                                return tmp13;
                            }
                            ;
                            auto tmp14 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp15 = static_cast<float>(0.0);
                            auto tmp16 = tmp7 ? tmp14 : tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp18 = static_cast<float>(0.0);
                        auto tmp19 = tmp3 ? tmp17 : tmp18;
                        auto tmp20 = decltype(tmp0)(tmp0 + tmp19);
                        in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp20;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr10 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(255L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(248L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr10 + static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(248L); x2<static_cast<long>(255L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr10[static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr10 + static_cast<long>(x0));
                tmp0.store(out_ptr11 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr11 + static_cast<long>(131328L + x2 + (513L*x1) + (525312L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr11 + static_cast<long>(x0));
                tmp0.store(out_ptr12 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr12 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr12[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x2);
                            auto tmp4 = static_cast<long>(255);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(x3);
                                auto tmp8 = static_cast<long>(258);
                                auto tmp9 = tmp7 >= tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_out_ptr0[static_cast<long>(256L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                auto tmp13 = static_cast<float>(0.0);
                                auto tmp14 = tmp9 ? tmp12 : tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp16 = static_cast<float>(0.0);
                            auto tmp17 = tmp5 ? tmp15 : tmp16;
                            auto tmp18 = tmp2 ? tmp17 : tmp16;
                            auto tmp19 = tmp3 >= tmp4;
                            auto tmp20 = static_cast<long>(511);
                            auto tmp21 = tmp3 < tmp20;
                            auto tmp22 = tmp19 & tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(x3);
                                auto tmp25 = static_cast<long>(257);
                                auto tmp26 = tmp24 >= tmp25;
                                auto tmp27 = [&]
                                {
                                    auto tmp28 = out_ptr10[static_cast<long>(256L + x3 + (513L*x2) + (131328L*x1) + (525312L*x0))];
                                    return tmp28;
                                }
                                ;
                                auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                                auto tmp30 = static_cast<float>(0.0);
                                auto tmp31 = tmp26 ? tmp29 : tmp30;
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp33 = tmp22 ? tmp32 : tmp16;
                            auto tmp34 = decltype(tmp18)(tmp18 + tmp33);
                            auto tmp35 = static_cast<int>(2);
                            auto tmp36 = tmp0 == tmp35;
                            auto tmp37 = static_cast<long>(256);
                            auto tmp38 = tmp3 >= tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x3);
                                auto tmp41 = static_cast<long>(257);
                                auto tmp42 = tmp40 < tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = out_ptr11[static_cast<long>(262912L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp44;
                                }
                                ;
                                auto tmp45 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp46 = static_cast<float>(0.0);
                                auto tmp47 = tmp42 ? tmp45 : tmp46;
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp38 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp49 = tmp38 ? tmp48 : tmp16;
                            auto tmp50 = tmp36 ? tmp49 : tmp16;
                            auto tmp51 = decltype(tmp34)(tmp34 + tmp50);
                            out_ptr13[static_cast<long>(x3 + (513L*x2) + (262656L*x1) + (787968L*x0))] = tmp51;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = static_cast<long>(513);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = out_ptr13[static_cast<long>(x3 + (512L*x2) + (262656L*x1) + (262656L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (787968L*x0) + (787968L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                auto tmp5 = c10::convert<long>(c10::div_floor_integer((x3 + (512L*x2)), 513L));
                                auto tmp6 = static_cast<long>(256);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x3 + (512L*x2))) % static_cast<long>(513L));
                                    auto tmp10 = static_cast<long>(257);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = out_ptr12[static_cast<long>(256L + x3 + (512L*x2) + (131328L*x1) + (131328L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (525312L*x0) + (525312L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                auto tmp20 = decltype(tmp4)(tmp4 + tmp19);
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                            out_ptr14[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (786432L*x0))] = tmp21;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_index_add_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_as_strided_scatter_div_view_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp3.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = c10::convert<int>(256L + x0);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<int>(1536);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = tmp2 & tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(16384L + (64L*x0) + (98304L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp5)); })();
                        return tmp7;
                    }
                    ;
                    auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_66 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_clone_sum_69 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_new_zeros_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1179648L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14)
{
    auto out_ptr8 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0)];
                            auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = static_cast<long>(770);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = [&]
                            {
                                auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                                auto tmp6 = static_cast<long>(196864);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                    auto tmp10 = static_cast<long>(768);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                            auto tmp22 = c10::convert<float>(tmp21);
                            auto tmp23 = static_cast<float>(1.1111111111111112);
                            auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                            auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                            auto tmp26 = static_cast<float>(0.0);
                            auto tmp27 = tmp0 ? tmp26 : tmp25;
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            tmp_acc0 = tmp_acc0 + tmp29;
                        }
                        out_ptr0[static_cast<long>(x1 + (12L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp30 = out_ptr0[static_cast<long>(x1 + (12L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = static_cast<long>(770);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                            auto tmp6 = static_cast<long>(196864);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                auto tmp10 = static_cast<long>(768);
                                auto tmp11 = tmp9 < tmp10;
                                auto tmp12 = [&]
                                {
                                    auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                    return tmp13;
                                }
                                ;
                                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                auto tmp15 = static_cast<float>(0.0);
                                auto tmp16 = tmp11 ? tmp14 : tmp15;
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp18 = static_cast<float>(0.0);
                            auto tmp19 = tmp7 ? tmp17 : tmp18;
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp22 = c10::convert<float>(tmp21);
                        auto tmp23 = static_cast<float>(1.1111111111111112);
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = static_cast<float>(0.0);
                        auto tmp27 = tmp0 ? tmp26 : tmp25;
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                        auto tmp32 = decltype(tmp29)(tmp29 - tmp31);
                        out_ptr1[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp32;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr5 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr5[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr4[static_cast<long>((-197632L) + x2 + (257L*x1))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr4[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp11 = static_cast<float>(0.0);
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = static_cast<float>(0.0);
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp17 = static_cast<float>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        out_ptr6[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp18;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr7 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                        tmp0.store(out_ptr7 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr6[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        out_ptr7[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(out_ptr8 + static_cast<long>(x0));
                tmp2.store(out_ptr9 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr9 + static_cast<long>(x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp1 = c10::convert<long>(x1);
                        auto tmp2 = static_cast<long>(256);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2);
                            auto tmp6 = static_cast<long>(257);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = in_ptr5[static_cast<long>(x2 + (257L*x1))];
                                auto tmp10 = c10::convert<bool>(tmp9);
                                auto tmp11 = out_ptr8[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp12 = static_cast<float>(0.0);
                                auto tmp13 = tmp10 ? tmp12 : tmp11;
                                return tmp13;
                            }
                            ;
                            auto tmp14 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp15 = static_cast<float>(0.0);
                            auto tmp16 = tmp7 ? tmp14 : tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp18 = static_cast<float>(0.0);
                        auto tmp19 = tmp3 ? tmp17 : tmp18;
                        auto tmp20 = decltype(tmp0)(tmp0 + tmp19);
                        in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp20;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr10 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(255L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(248L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr10 + static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(248L); x2<static_cast<long>(255L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr10[static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr10 + static_cast<long>(x0));
                tmp0.store(out_ptr11 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr11 + static_cast<long>(131328L + x2 + (513L*x1) + (525312L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr11 + static_cast<long>(x0));
                tmp0.store(out_ptr12 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr12 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr12[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x2);
                            auto tmp4 = static_cast<long>(255);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(x3);
                                auto tmp8 = static_cast<long>(258);
                                auto tmp9 = tmp7 >= tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_out_ptr0[static_cast<long>(256L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                auto tmp13 = static_cast<float>(0.0);
                                auto tmp14 = tmp9 ? tmp12 : tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp16 = static_cast<float>(0.0);
                            auto tmp17 = tmp5 ? tmp15 : tmp16;
                            auto tmp18 = tmp2 ? tmp17 : tmp16;
                            auto tmp19 = tmp3 >= tmp4;
                            auto tmp20 = static_cast<long>(511);
                            auto tmp21 = tmp3 < tmp20;
                            auto tmp22 = tmp19 & tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(x3);
                                auto tmp25 = static_cast<long>(257);
                                auto tmp26 = tmp24 >= tmp25;
                                auto tmp27 = [&]
                                {
                                    auto tmp28 = out_ptr10[static_cast<long>(256L + x3 + (513L*x2) + (131328L*x1) + (525312L*x0))];
                                    return tmp28;
                                }
                                ;
                                auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                                auto tmp30 = static_cast<float>(0.0);
                                auto tmp31 = tmp26 ? tmp29 : tmp30;
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp33 = tmp22 ? tmp32 : tmp16;
                            auto tmp34 = decltype(tmp18)(tmp18 + tmp33);
                            auto tmp35 = static_cast<int>(2);
                            auto tmp36 = tmp0 == tmp35;
                            auto tmp37 = static_cast<long>(256);
                            auto tmp38 = tmp3 >= tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x3);
                                auto tmp41 = static_cast<long>(257);
                                auto tmp42 = tmp40 < tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = out_ptr11[static_cast<long>(262912L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp44;
                                }
                                ;
                                auto tmp45 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp46 = static_cast<float>(0.0);
                                auto tmp47 = tmp42 ? tmp45 : tmp46;
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp38 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp49 = tmp38 ? tmp48 : tmp16;
                            auto tmp50 = tmp36 ? tmp49 : tmp16;
                            auto tmp51 = decltype(tmp34)(tmp34 + tmp50);
                            out_ptr13[static_cast<long>(x3 + (513L*x2) + (262656L*x1) + (787968L*x0))] = tmp51;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = static_cast<long>(513);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = out_ptr13[static_cast<long>(x3 + (512L*x2) + (262656L*x1) + (262656L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (787968L*x0) + (787968L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                auto tmp5 = c10::convert<long>(c10::div_floor_integer((x3 + (512L*x2)), 513L));
                                auto tmp6 = static_cast<long>(256);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x3 + (512L*x2))) % static_cast<long>(513L));
                                    auto tmp10 = static_cast<long>(257);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = out_ptr12[static_cast<long>(256L + x3 + (512L*x2) + (131328L*x1) + (131328L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (525312L*x0) + (525312L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                auto tmp20 = decltype(tmp4)(tmp4 + tmp19);
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                            out_ptr14[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (786432L*x0))] = tmp21;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_index_add_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_as_strided_scatter_div_view_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp3.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = c10::convert<int>(256L + x0);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<int>(1536);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = tmp2 & tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(16384L + (64L*x0) + (98304L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp5)); })();
                        return tmp7;
                    }
                    ;
                    auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_77 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_78 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_clone_sum_80 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_new_zeros_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1179648L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14)
{
    auto out_ptr8 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0)];
                            auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = static_cast<long>(770);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = [&]
                            {
                                auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                                auto tmp6 = static_cast<long>(196864);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                    auto tmp10 = static_cast<long>(768);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                            auto tmp22 = c10::convert<float>(tmp21);
                            auto tmp23 = static_cast<float>(1.1111111111111112);
                            auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                            auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                            auto tmp26 = static_cast<float>(0.0);
                            auto tmp27 = tmp0 ? tmp26 : tmp25;
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            tmp_acc0 = tmp_acc0 + tmp29;
                        }
                        out_ptr0[static_cast<long>(x1 + (12L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp30 = out_ptr0[static_cast<long>(x1 + (12L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = static_cast<long>(770);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                            auto tmp6 = static_cast<long>(196864);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                auto tmp10 = static_cast<long>(768);
                                auto tmp11 = tmp9 < tmp10;
                                auto tmp12 = [&]
                                {
                                    auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                    return tmp13;
                                }
                                ;
                                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                auto tmp15 = static_cast<float>(0.0);
                                auto tmp16 = tmp11 ? tmp14 : tmp15;
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp18 = static_cast<float>(0.0);
                            auto tmp19 = tmp7 ? tmp17 : tmp18;
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp22 = c10::convert<float>(tmp21);
                        auto tmp23 = static_cast<float>(1.1111111111111112);
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = static_cast<float>(0.0);
                        auto tmp27 = tmp0 ? tmp26 : tmp25;
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                        auto tmp32 = decltype(tmp29)(tmp29 - tmp31);
                        out_ptr1[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp32;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr5 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr5[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr4[static_cast<long>((-197632L) + x2 + (257L*x1))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr4[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp11 = static_cast<float>(0.0);
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = static_cast<float>(0.0);
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp17 = static_cast<float>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        out_ptr6[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp18;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr7 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                        tmp0.store(out_ptr7 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr6[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        out_ptr7[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(out_ptr8 + static_cast<long>(x0));
                tmp2.store(out_ptr9 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr9 + static_cast<long>(x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp1 = c10::convert<long>(x1);
                        auto tmp2 = static_cast<long>(256);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2);
                            auto tmp6 = static_cast<long>(257);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = in_ptr5[static_cast<long>(x2 + (257L*x1))];
                                auto tmp10 = c10::convert<bool>(tmp9);
                                auto tmp11 = out_ptr8[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp12 = static_cast<float>(0.0);
                                auto tmp13 = tmp10 ? tmp12 : tmp11;
                                return tmp13;
                            }
                            ;
                            auto tmp14 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp15 = static_cast<float>(0.0);
                            auto tmp16 = tmp7 ? tmp14 : tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp18 = static_cast<float>(0.0);
                        auto tmp19 = tmp3 ? tmp17 : tmp18;
                        auto tmp20 = decltype(tmp0)(tmp0 + tmp19);
                        in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp20;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr10 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(255L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(248L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr10 + static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(248L); x2<static_cast<long>(255L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr10[static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr10 + static_cast<long>(x0));
                tmp0.store(out_ptr11 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr11 + static_cast<long>(131328L + x2 + (513L*x1) + (525312L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr11 + static_cast<long>(x0));
                tmp0.store(out_ptr12 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr12 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr12[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x2);
                            auto tmp4 = static_cast<long>(255);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(x3);
                                auto tmp8 = static_cast<long>(258);
                                auto tmp9 = tmp7 >= tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_out_ptr0[static_cast<long>(256L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                auto tmp13 = static_cast<float>(0.0);
                                auto tmp14 = tmp9 ? tmp12 : tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp16 = static_cast<float>(0.0);
                            auto tmp17 = tmp5 ? tmp15 : tmp16;
                            auto tmp18 = tmp2 ? tmp17 : tmp16;
                            auto tmp19 = tmp3 >= tmp4;
                            auto tmp20 = static_cast<long>(511);
                            auto tmp21 = tmp3 < tmp20;
                            auto tmp22 = tmp19 & tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(x3);
                                auto tmp25 = static_cast<long>(257);
                                auto tmp26 = tmp24 >= tmp25;
                                auto tmp27 = [&]
                                {
                                    auto tmp28 = out_ptr10[static_cast<long>(256L + x3 + (513L*x2) + (131328L*x1) + (525312L*x0))];
                                    return tmp28;
                                }
                                ;
                                auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                                auto tmp30 = static_cast<float>(0.0);
                                auto tmp31 = tmp26 ? tmp29 : tmp30;
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp33 = tmp22 ? tmp32 : tmp16;
                            auto tmp34 = decltype(tmp18)(tmp18 + tmp33);
                            auto tmp35 = static_cast<int>(2);
                            auto tmp36 = tmp0 == tmp35;
                            auto tmp37 = static_cast<long>(256);
                            auto tmp38 = tmp3 >= tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x3);
                                auto tmp41 = static_cast<long>(257);
                                auto tmp42 = tmp40 < tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = out_ptr11[static_cast<long>(262912L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp44;
                                }
                                ;
                                auto tmp45 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp46 = static_cast<float>(0.0);
                                auto tmp47 = tmp42 ? tmp45 : tmp46;
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp38 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp49 = tmp38 ? tmp48 : tmp16;
                            auto tmp50 = tmp36 ? tmp49 : tmp16;
                            auto tmp51 = decltype(tmp34)(tmp34 + tmp50);
                            out_ptr13[static_cast<long>(x3 + (513L*x2) + (262656L*x1) + (787968L*x0))] = tmp51;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = static_cast<long>(513);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = out_ptr13[static_cast<long>(x3 + (512L*x2) + (262656L*x1) + (262656L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (787968L*x0) + (787968L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                auto tmp5 = c10::convert<long>(c10::div_floor_integer((x3 + (512L*x2)), 513L));
                                auto tmp6 = static_cast<long>(256);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x3 + (512L*x2))) % static_cast<long>(513L));
                                    auto tmp10 = static_cast<long>(257);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = out_ptr12[static_cast<long>(256L + x3 + (512L*x2) + (131328L*x1) + (131328L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (525312L*x0) + (525312L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                auto tmp20 = decltype(tmp4)(tmp4 + tmp19);
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                            out_ptr14[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (786432L*x0))] = tmp21;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_index_add_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_as_strided_scatter_div_view_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp3.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = c10::convert<int>(256L + x0);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<int>(1536);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = tmp2 & tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(16384L + (64L*x0) + (98304L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp5)); })();
                        return tmp7;
                    }
                    ;
                    auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_88 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_clone_sum_91 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_new_zeros_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1179648L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14)
{
    auto out_ptr8 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0)];
                            auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = static_cast<long>(770);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = [&]
                            {
                                auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                                auto tmp6 = static_cast<long>(196864);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                    auto tmp10 = static_cast<long>(768);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                            auto tmp22 = c10::convert<float>(tmp21);
                            auto tmp23 = static_cast<float>(1.1111111111111112);
                            auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                            auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                            auto tmp26 = static_cast<float>(0.0);
                            auto tmp27 = tmp0 ? tmp26 : tmp25;
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            tmp_acc0 = tmp_acc0 + tmp29;
                        }
                        out_ptr0[static_cast<long>(x1 + (12L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp30 = out_ptr0[static_cast<long>(x1 + (12L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = static_cast<long>(770);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                            auto tmp6 = static_cast<long>(196864);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                auto tmp10 = static_cast<long>(768);
                                auto tmp11 = tmp9 < tmp10;
                                auto tmp12 = [&]
                                {
                                    auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                    return tmp13;
                                }
                                ;
                                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                auto tmp15 = static_cast<float>(0.0);
                                auto tmp16 = tmp11 ? tmp14 : tmp15;
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp18 = static_cast<float>(0.0);
                            auto tmp19 = tmp7 ? tmp17 : tmp18;
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp22 = c10::convert<float>(tmp21);
                        auto tmp23 = static_cast<float>(1.1111111111111112);
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = static_cast<float>(0.0);
                        auto tmp27 = tmp0 ? tmp26 : tmp25;
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                        auto tmp32 = decltype(tmp29)(tmp29 - tmp31);
                        out_ptr1[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp32;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr5 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr5[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr4[static_cast<long>((-197632L) + x2 + (257L*x1))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr4[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp11 = static_cast<float>(0.0);
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = static_cast<float>(0.0);
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp17 = static_cast<float>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        out_ptr6[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp18;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr7 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                        tmp0.store(out_ptr7 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr6[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        out_ptr7[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(out_ptr8 + static_cast<long>(x0));
                tmp2.store(out_ptr9 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr9 + static_cast<long>(x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp1 = c10::convert<long>(x1);
                        auto tmp2 = static_cast<long>(256);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2);
                            auto tmp6 = static_cast<long>(257);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = in_ptr5[static_cast<long>(x2 + (257L*x1))];
                                auto tmp10 = c10::convert<bool>(tmp9);
                                auto tmp11 = out_ptr8[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp12 = static_cast<float>(0.0);
                                auto tmp13 = tmp10 ? tmp12 : tmp11;
                                return tmp13;
                            }
                            ;
                            auto tmp14 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp15 = static_cast<float>(0.0);
                            auto tmp16 = tmp7 ? tmp14 : tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp18 = static_cast<float>(0.0);
                        auto tmp19 = tmp3 ? tmp17 : tmp18;
                        auto tmp20 = decltype(tmp0)(tmp0 + tmp19);
                        in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp20;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr10 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(255L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(248L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr10 + static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(248L); x2<static_cast<long>(255L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr10[static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr10 + static_cast<long>(x0));
                tmp0.store(out_ptr11 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr11 + static_cast<long>(131328L + x2 + (513L*x1) + (525312L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr11 + static_cast<long>(x0));
                tmp0.store(out_ptr12 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr12 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr12[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x2);
                            auto tmp4 = static_cast<long>(255);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(x3);
                                auto tmp8 = static_cast<long>(258);
                                auto tmp9 = tmp7 >= tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_out_ptr0[static_cast<long>(256L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                auto tmp13 = static_cast<float>(0.0);
                                auto tmp14 = tmp9 ? tmp12 : tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp16 = static_cast<float>(0.0);
                            auto tmp17 = tmp5 ? tmp15 : tmp16;
                            auto tmp18 = tmp2 ? tmp17 : tmp16;
                            auto tmp19 = tmp3 >= tmp4;
                            auto tmp20 = static_cast<long>(511);
                            auto tmp21 = tmp3 < tmp20;
                            auto tmp22 = tmp19 & tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(x3);
                                auto tmp25 = static_cast<long>(257);
                                auto tmp26 = tmp24 >= tmp25;
                                auto tmp27 = [&]
                                {
                                    auto tmp28 = out_ptr10[static_cast<long>(256L + x3 + (513L*x2) + (131328L*x1) + (525312L*x0))];
                                    return tmp28;
                                }
                                ;
                                auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                                auto tmp30 = static_cast<float>(0.0);
                                auto tmp31 = tmp26 ? tmp29 : tmp30;
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp33 = tmp22 ? tmp32 : tmp16;
                            auto tmp34 = decltype(tmp18)(tmp18 + tmp33);
                            auto tmp35 = static_cast<int>(2);
                            auto tmp36 = tmp0 == tmp35;
                            auto tmp37 = static_cast<long>(256);
                            auto tmp38 = tmp3 >= tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x3);
                                auto tmp41 = static_cast<long>(257);
                                auto tmp42 = tmp40 < tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = out_ptr11[static_cast<long>(262912L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp44;
                                }
                                ;
                                auto tmp45 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp46 = static_cast<float>(0.0);
                                auto tmp47 = tmp42 ? tmp45 : tmp46;
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp38 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp49 = tmp38 ? tmp48 : tmp16;
                            auto tmp50 = tmp36 ? tmp49 : tmp16;
                            auto tmp51 = decltype(tmp34)(tmp34 + tmp50);
                            out_ptr13[static_cast<long>(x3 + (513L*x2) + (262656L*x1) + (787968L*x0))] = tmp51;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = static_cast<long>(513);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = out_ptr13[static_cast<long>(x3 + (512L*x2) + (262656L*x1) + (262656L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (787968L*x0) + (787968L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                auto tmp5 = c10::convert<long>(c10::div_floor_integer((x3 + (512L*x2)), 513L));
                                auto tmp6 = static_cast<long>(256);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x3 + (512L*x2))) % static_cast<long>(513L));
                                    auto tmp10 = static_cast<long>(257);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = out_ptr12[static_cast<long>(256L + x3 + (512L*x2) + (131328L*x1) + (131328L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (525312L*x0) + (525312L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                auto tmp20 = decltype(tmp4)(tmp4 + tmp19);
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                            out_ptr14[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (786432L*x0))] = tmp21;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_index_add_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_as_strided_scatter_div_view_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp3.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = c10::convert<int>(256L + x0);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<int>(1536);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = tmp2 & tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(16384L + (64L*x0) + (98304L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp5)); })();
                        return tmp7;
                    }
                    ;
                    auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_99 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_100 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_clone_sum_102 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_new_zeros_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1179648L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14)
{
    auto out_ptr8 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0)];
                            auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = static_cast<long>(770);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = [&]
                            {
                                auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                                auto tmp6 = static_cast<long>(196864);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                    auto tmp10 = static_cast<long>(768);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                            auto tmp22 = c10::convert<float>(tmp21);
                            auto tmp23 = static_cast<float>(1.1111111111111112);
                            auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                            auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                            auto tmp26 = static_cast<float>(0.0);
                            auto tmp27 = tmp0 ? tmp26 : tmp25;
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            tmp_acc0 = tmp_acc0 + tmp29;
                        }
                        out_ptr0[static_cast<long>(x1 + (12L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp30 = out_ptr0[static_cast<long>(x1 + (12L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = static_cast<long>(770);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                            auto tmp6 = static_cast<long>(196864);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                auto tmp10 = static_cast<long>(768);
                                auto tmp11 = tmp9 < tmp10;
                                auto tmp12 = [&]
                                {
                                    auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                    return tmp13;
                                }
                                ;
                                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                auto tmp15 = static_cast<float>(0.0);
                                auto tmp16 = tmp11 ? tmp14 : tmp15;
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp18 = static_cast<float>(0.0);
                            auto tmp19 = tmp7 ? tmp17 : tmp18;
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp22 = c10::convert<float>(tmp21);
                        auto tmp23 = static_cast<float>(1.1111111111111112);
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = static_cast<float>(0.0);
                        auto tmp27 = tmp0 ? tmp26 : tmp25;
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                        auto tmp32 = decltype(tmp29)(tmp29 - tmp31);
                        out_ptr1[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp32;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr5 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr5[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr4[static_cast<long>((-197632L) + x2 + (257L*x1))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr4[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp11 = static_cast<float>(0.0);
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = static_cast<float>(0.0);
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp17 = static_cast<float>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        out_ptr6[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp18;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr7 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                        tmp0.store(out_ptr7 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr6[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        out_ptr7[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(out_ptr8 + static_cast<long>(x0));
                tmp2.store(out_ptr9 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr9 + static_cast<long>(x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp1 = c10::convert<long>(x1);
                        auto tmp2 = static_cast<long>(256);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2);
                            auto tmp6 = static_cast<long>(257);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = in_ptr5[static_cast<long>(x2 + (257L*x1))];
                                auto tmp10 = c10::convert<bool>(tmp9);
                                auto tmp11 = out_ptr8[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp12 = static_cast<float>(0.0);
                                auto tmp13 = tmp10 ? tmp12 : tmp11;
                                return tmp13;
                            }
                            ;
                            auto tmp14 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp15 = static_cast<float>(0.0);
                            auto tmp16 = tmp7 ? tmp14 : tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp18 = static_cast<float>(0.0);
                        auto tmp19 = tmp3 ? tmp17 : tmp18;
                        auto tmp20 = decltype(tmp0)(tmp0 + tmp19);
                        in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp20;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr10 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(255L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(248L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr10 + static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(248L); x2<static_cast<long>(255L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr10[static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr10 + static_cast<long>(x0));
                tmp0.store(out_ptr11 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr11 + static_cast<long>(131328L + x2 + (513L*x1) + (525312L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr11 + static_cast<long>(x0));
                tmp0.store(out_ptr12 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr12 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr12[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x2);
                            auto tmp4 = static_cast<long>(255);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(x3);
                                auto tmp8 = static_cast<long>(258);
                                auto tmp9 = tmp7 >= tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_out_ptr0[static_cast<long>(256L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                auto tmp13 = static_cast<float>(0.0);
                                auto tmp14 = tmp9 ? tmp12 : tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp16 = static_cast<float>(0.0);
                            auto tmp17 = tmp5 ? tmp15 : tmp16;
                            auto tmp18 = tmp2 ? tmp17 : tmp16;
                            auto tmp19 = tmp3 >= tmp4;
                            auto tmp20 = static_cast<long>(511);
                            auto tmp21 = tmp3 < tmp20;
                            auto tmp22 = tmp19 & tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(x3);
                                auto tmp25 = static_cast<long>(257);
                                auto tmp26 = tmp24 >= tmp25;
                                auto tmp27 = [&]
                                {
                                    auto tmp28 = out_ptr10[static_cast<long>(256L + x3 + (513L*x2) + (131328L*x1) + (525312L*x0))];
                                    return tmp28;
                                }
                                ;
                                auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                                auto tmp30 = static_cast<float>(0.0);
                                auto tmp31 = tmp26 ? tmp29 : tmp30;
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp33 = tmp22 ? tmp32 : tmp16;
                            auto tmp34 = decltype(tmp18)(tmp18 + tmp33);
                            auto tmp35 = static_cast<int>(2);
                            auto tmp36 = tmp0 == tmp35;
                            auto tmp37 = static_cast<long>(256);
                            auto tmp38 = tmp3 >= tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x3);
                                auto tmp41 = static_cast<long>(257);
                                auto tmp42 = tmp40 < tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = out_ptr11[static_cast<long>(262912L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp44;
                                }
                                ;
                                auto tmp45 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp46 = static_cast<float>(0.0);
                                auto tmp47 = tmp42 ? tmp45 : tmp46;
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp38 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp49 = tmp38 ? tmp48 : tmp16;
                            auto tmp50 = tmp36 ? tmp49 : tmp16;
                            auto tmp51 = decltype(tmp34)(tmp34 + tmp50);
                            out_ptr13[static_cast<long>(x3 + (513L*x2) + (262656L*x1) + (787968L*x0))] = tmp51;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = static_cast<long>(513);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = out_ptr13[static_cast<long>(x3 + (512L*x2) + (262656L*x1) + (262656L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (787968L*x0) + (787968L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                auto tmp5 = c10::convert<long>(c10::div_floor_integer((x3 + (512L*x2)), 513L));
                                auto tmp6 = static_cast<long>(256);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x3 + (512L*x2))) % static_cast<long>(513L));
                                    auto tmp10 = static_cast<long>(257);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = out_ptr12[static_cast<long>(256L + x3 + (512L*x2) + (131328L*x1) + (131328L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (525312L*x0) + (525312L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                auto tmp20 = decltype(tmp4)(tmp4 + tmp19);
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                            out_ptr14[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (786432L*x0))] = tmp21;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_index_add_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_as_strided_scatter_div_view_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp3.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = c10::convert<int>(256L + x0);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<int>(1536);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = tmp2 & tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(16384L + (64L*x0) + (98304L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp5)); })();
                        return tmp7;
                    }
                    ;
                    auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_110 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_111 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_clone_sum_113 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_new_zeros_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1179648L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_115 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14)
{
    auto out_ptr8 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0)];
                            auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = static_cast<long>(770);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = [&]
                            {
                                auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                                auto tmp6 = static_cast<long>(196864);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                    auto tmp10 = static_cast<long>(768);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                            auto tmp22 = c10::convert<float>(tmp21);
                            auto tmp23 = static_cast<float>(1.1111111111111112);
                            auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                            auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                            auto tmp26 = static_cast<float>(0.0);
                            auto tmp27 = tmp0 ? tmp26 : tmp25;
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            tmp_acc0 = tmp_acc0 + tmp29;
                        }
                        out_ptr0[static_cast<long>(x1 + (12L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp30 = out_ptr0[static_cast<long>(x1 + (12L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = static_cast<long>(770);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                            auto tmp6 = static_cast<long>(196864);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                auto tmp10 = static_cast<long>(768);
                                auto tmp11 = tmp9 < tmp10;
                                auto tmp12 = [&]
                                {
                                    auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                    return tmp13;
                                }
                                ;
                                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                auto tmp15 = static_cast<float>(0.0);
                                auto tmp16 = tmp11 ? tmp14 : tmp15;
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp18 = static_cast<float>(0.0);
                            auto tmp19 = tmp7 ? tmp17 : tmp18;
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp22 = c10::convert<float>(tmp21);
                        auto tmp23 = static_cast<float>(1.1111111111111112);
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = static_cast<float>(0.0);
                        auto tmp27 = tmp0 ? tmp26 : tmp25;
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                        auto tmp32 = decltype(tmp29)(tmp29 - tmp31);
                        out_ptr1[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp32;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr5 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr5[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr4[static_cast<long>((-197632L) + x2 + (257L*x1))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr4[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp11 = static_cast<float>(0.0);
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = static_cast<float>(0.0);
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp17 = static_cast<float>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        out_ptr6[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp18;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr7 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                        tmp0.store(out_ptr7 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr6[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        out_ptr7[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(out_ptr8 + static_cast<long>(x0));
                tmp2.store(out_ptr9 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr9 + static_cast<long>(x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp1 = c10::convert<long>(x1);
                        auto tmp2 = static_cast<long>(256);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2);
                            auto tmp6 = static_cast<long>(257);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = in_ptr5[static_cast<long>(x2 + (257L*x1))];
                                auto tmp10 = c10::convert<bool>(tmp9);
                                auto tmp11 = out_ptr8[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp12 = static_cast<float>(0.0);
                                auto tmp13 = tmp10 ? tmp12 : tmp11;
                                return tmp13;
                            }
                            ;
                            auto tmp14 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp15 = static_cast<float>(0.0);
                            auto tmp16 = tmp7 ? tmp14 : tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp18 = static_cast<float>(0.0);
                        auto tmp19 = tmp3 ? tmp17 : tmp18;
                        auto tmp20 = decltype(tmp0)(tmp0 + tmp19);
                        in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp20;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr10 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(255L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(248L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr10 + static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(248L); x2<static_cast<long>(255L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr10[static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr10 + static_cast<long>(x0));
                tmp0.store(out_ptr11 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr11 + static_cast<long>(131328L + x2 + (513L*x1) + (525312L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr11 + static_cast<long>(x0));
                tmp0.store(out_ptr12 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr12 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr12[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x2);
                            auto tmp4 = static_cast<long>(255);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(x3);
                                auto tmp8 = static_cast<long>(258);
                                auto tmp9 = tmp7 >= tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_out_ptr0[static_cast<long>(256L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                auto tmp13 = static_cast<float>(0.0);
                                auto tmp14 = tmp9 ? tmp12 : tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp16 = static_cast<float>(0.0);
                            auto tmp17 = tmp5 ? tmp15 : tmp16;
                            auto tmp18 = tmp2 ? tmp17 : tmp16;
                            auto tmp19 = tmp3 >= tmp4;
                            auto tmp20 = static_cast<long>(511);
                            auto tmp21 = tmp3 < tmp20;
                            auto tmp22 = tmp19 & tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(x3);
                                auto tmp25 = static_cast<long>(257);
                                auto tmp26 = tmp24 >= tmp25;
                                auto tmp27 = [&]
                                {
                                    auto tmp28 = out_ptr10[static_cast<long>(256L + x3 + (513L*x2) + (131328L*x1) + (525312L*x0))];
                                    return tmp28;
                                }
                                ;
                                auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                                auto tmp30 = static_cast<float>(0.0);
                                auto tmp31 = tmp26 ? tmp29 : tmp30;
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp33 = tmp22 ? tmp32 : tmp16;
                            auto tmp34 = decltype(tmp18)(tmp18 + tmp33);
                            auto tmp35 = static_cast<int>(2);
                            auto tmp36 = tmp0 == tmp35;
                            auto tmp37 = static_cast<long>(256);
                            auto tmp38 = tmp3 >= tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x3);
                                auto tmp41 = static_cast<long>(257);
                                auto tmp42 = tmp40 < tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = out_ptr11[static_cast<long>(262912L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp44;
                                }
                                ;
                                auto tmp45 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp46 = static_cast<float>(0.0);
                                auto tmp47 = tmp42 ? tmp45 : tmp46;
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp38 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp49 = tmp38 ? tmp48 : tmp16;
                            auto tmp50 = tmp36 ? tmp49 : tmp16;
                            auto tmp51 = decltype(tmp34)(tmp34 + tmp50);
                            out_ptr13[static_cast<long>(x3 + (513L*x2) + (262656L*x1) + (787968L*x0))] = tmp51;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = static_cast<long>(513);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = out_ptr13[static_cast<long>(x3 + (512L*x2) + (262656L*x1) + (262656L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (787968L*x0) + (787968L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                auto tmp5 = c10::convert<long>(c10::div_floor_integer((x3 + (512L*x2)), 513L));
                                auto tmp6 = static_cast<long>(256);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x3 + (512L*x2))) % static_cast<long>(513L));
                                    auto tmp10 = static_cast<long>(257);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = out_ptr12[static_cast<long>(256L + x3 + (512L*x2) + (131328L*x1) + (131328L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (525312L*x0) + (525312L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                auto tmp20 = decltype(tmp4)(tmp4 + tmp19);
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                            out_ptr14[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (786432L*x0))] = tmp21;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_index_add_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_as_strided_scatter_div_view_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp3.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = c10::convert<int>(256L + x0);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<int>(1536);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = tmp2 & tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(16384L + (64L*x0) + (98304L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp5)); })();
                        return tmp7;
                    }
                    ;
                    auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_121 = async_compile.cpp('''
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
                       const bool* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr1 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr8[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_122 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const bool* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                out_ptr6[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_clone_sum_124 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (768L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_new_zeros_125 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1179648L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_126 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14)
{
    auto out_ptr8 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0)];
                            auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = static_cast<long>(770);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = [&]
                            {
                                auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                                auto tmp6 = static_cast<long>(196864);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                    auto tmp10 = static_cast<long>(768);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                return tmp19;
                            }
                            ;
                            auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                            auto tmp22 = c10::convert<float>(tmp21);
                            auto tmp23 = static_cast<float>(1.1111111111111112);
                            auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                            auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                            auto tmp26 = static_cast<float>(0.0);
                            auto tmp27 = tmp0 ? tmp26 : tmp25;
                            auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                            tmp_acc0 = tmp_acc0 + tmp29;
                        }
                        out_ptr0[static_cast<long>(x1 + (12L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp21 = in_ptr2[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp28 = in_ptr3[static_cast<long>(x2 + (513L*x1) + (6156L*x0))];
                        auto tmp30 = out_ptr0[static_cast<long>(x1 + (12L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = static_cast<long>(770);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))));
                            auto tmp6 = static_cast<long>(196864);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = c10::convert<long>(static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L));
                                auto tmp10 = static_cast<long>(768);
                                auto tmp11 = tmp9 < tmp10;
                                auto tmp12 = [&]
                                {
                                    auto tmp13 = in_ptr1[static_cast<long>((768L*(static_cast<long>(c10::div_floor_integer((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L)))), 769L)) % static_cast<long>(256L))) + (196608L*(c10::div_floor_integer(x0, 256L))) + (786432L*x1) + (static_cast<long>((x2 + (770L*(static_cast<long>(x0) % static_cast<long>(256L))))) % static_cast<long>(769L)))];
                                    return tmp13;
                                }
                                ;
                                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                auto tmp15 = static_cast<float>(0.0);
                                auto tmp16 = tmp11 ? tmp14 : tmp15;
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp18 = static_cast<float>(0.0);
                            auto tmp19 = tmp7 ? tmp17 : tmp18;
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp22 = c10::convert<float>(tmp21);
                        auto tmp23 = static_cast<float>(1.1111111111111112);
                        auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                        auto tmp25 = decltype(tmp20)(tmp20 * tmp24);
                        auto tmp26 = static_cast<float>(0.0);
                        auto tmp27 = tmp0 ? tmp26 : tmp25;
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp31 = decltype(tmp28)(tmp28 * tmp30);
                        auto tmp32 = decltype(tmp29)(tmp29 - tmp31);
                        out_ptr1[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp32;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
                tmp0.store(out_ptr5 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr5 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr5[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(768);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = c10::convert<long>(x2);
                            auto tmp5 = static_cast<long>(256);
                            auto tmp6 = tmp4 >= tmp5;
                            auto tmp7 = [&]
                            {
                                auto tmp8 = in_ptr4[static_cast<long>((-197632L) + x2 + (257L*x1))];
                                auto tmp9 = c10::convert<bool>(tmp8);
                                auto tmp10 = out_ptr4[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp11 = static_cast<float>(0.0);
                                auto tmp12 = tmp9 ? tmp11 : tmp10;
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp6 ? tmp7() : static_cast<decltype(tmp7())>(0.0);
                            auto tmp14 = static_cast<float>(0.0);
                            auto tmp15 = tmp6 ? tmp13 : tmp14;
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp17 = static_cast<float>(0.0);
                        auto tmp18 = tmp2 ? tmp16 : tmp17;
                        out_ptr6[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp18;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr7 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                        tmp0.store(out_ptr7 + static_cast<long>(x2 + (513L*x0) + (525312L*x1)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(512L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr6[static_cast<long>(x2 + (513L*x0) + (525312L*x1))];
                        out_ptr7[static_cast<long>(x2 + (513L*x0) + (525312L*x1))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(out_ptr8 + static_cast<long>(x0));
                tmp2.store(out_ptr9 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr9 + static_cast<long>(x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(513L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr9[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                        auto tmp1 = c10::convert<long>(x1);
                        auto tmp2 = static_cast<long>(256);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = [&]
                        {
                            auto tmp5 = c10::convert<long>(x2);
                            auto tmp6 = static_cast<long>(257);
                            auto tmp7 = tmp5 < tmp6;
                            auto tmp8 = [&]
                            {
                                auto tmp9 = in_ptr5[static_cast<long>(x2 + (257L*x1))];
                                auto tmp10 = c10::convert<bool>(tmp9);
                                auto tmp11 = out_ptr8[static_cast<long>(x2 + (513L*x1) + (525312L*x0))];
                                auto tmp12 = static_cast<float>(0.0);
                                auto tmp13 = tmp10 ? tmp12 : tmp11;
                                return tmp13;
                            }
                            ;
                            auto tmp14 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                            auto tmp15 = static_cast<float>(0.0);
                            auto tmp16 = tmp7 ? tmp14 : tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = tmp3 ? tmp4() : static_cast<decltype(tmp4())>(0.0);
                        auto tmp18 = static_cast<float>(0.0);
                        auto tmp19 = tmp3 ? tmp17 : tmp18;
                        auto tmp20 = decltype(tmp0)(tmp0 + tmp19);
                        in_out_ptr0[static_cast<long>(x2 + (513L*x1) + (525312L*x0))] = tmp20;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr10 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(255L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(248L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr10 + static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(248L); x2<static_cast<long>(255L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr10[static_cast<long>(514L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr10 + static_cast<long>(x0));
                tmp0.store(out_ptr11 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr11 + static_cast<long>(131328L + x2 + (513L*x1) + (525312L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6303744L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr11 + static_cast<long>(x0));
                tmp0.store(out_ptr12 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr12 + static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(256L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        out_ptr12[static_cast<long>(394240L + x2 + (513L*x1) + (525312L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(513L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp3 = c10::convert<long>(x2);
                            auto tmp4 = static_cast<long>(255);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = c10::convert<long>(x3);
                                auto tmp8 = static_cast<long>(258);
                                auto tmp9 = tmp7 >= tmp8;
                                auto tmp10 = [&]
                                {
                                    auto tmp11 = in_out_ptr0[static_cast<long>(256L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp11;
                                }
                                ;
                                auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                                auto tmp13 = static_cast<float>(0.0);
                                auto tmp14 = tmp9 ? tmp12 : tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp16 = static_cast<float>(0.0);
                            auto tmp17 = tmp5 ? tmp15 : tmp16;
                            auto tmp18 = tmp2 ? tmp17 : tmp16;
                            auto tmp19 = tmp3 >= tmp4;
                            auto tmp20 = static_cast<long>(511);
                            auto tmp21 = tmp3 < tmp20;
                            auto tmp22 = tmp19 & tmp21;
                            auto tmp23 = [&]
                            {
                                auto tmp24 = c10::convert<long>(x3);
                                auto tmp25 = static_cast<long>(257);
                                auto tmp26 = tmp24 >= tmp25;
                                auto tmp27 = [&]
                                {
                                    auto tmp28 = out_ptr10[static_cast<long>(256L + x3 + (513L*x2) + (131328L*x1) + (525312L*x0))];
                                    return tmp28;
                                }
                                ;
                                auto tmp29 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                                auto tmp30 = static_cast<float>(0.0);
                                auto tmp31 = tmp26 ? tmp29 : tmp30;
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                            auto tmp33 = tmp22 ? tmp32 : tmp16;
                            auto tmp34 = decltype(tmp18)(tmp18 + tmp33);
                            auto tmp35 = static_cast<int>(2);
                            auto tmp36 = tmp0 == tmp35;
                            auto tmp37 = static_cast<long>(256);
                            auto tmp38 = tmp3 >= tmp37;
                            auto tmp39 = [&]
                            {
                                auto tmp40 = c10::convert<long>(x3);
                                auto tmp41 = static_cast<long>(257);
                                auto tmp42 = tmp40 < tmp41;
                                auto tmp43 = [&]
                                {
                                    auto tmp44 = out_ptr11[static_cast<long>(262912L + x3 + (513L*x2) + (525312L*x0))];
                                    return tmp44;
                                }
                                ;
                                auto tmp45 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                                auto tmp46 = static_cast<float>(0.0);
                                auto tmp47 = tmp42 ? tmp45 : tmp46;
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp38 ? tmp39() : static_cast<decltype(tmp39())>(0.0);
                            auto tmp49 = tmp38 ? tmp48 : tmp16;
                            auto tmp50 = tmp36 ? tmp49 : tmp16;
                            auto tmp51 = decltype(tmp34)(tmp34 + tmp50);
                            out_ptr13[static_cast<long>(x3 + (513L*x2) + (262656L*x1) + (787968L*x0))] = tmp51;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = static_cast<long>(513);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = out_ptr13[static_cast<long>(x3 + (512L*x2) + (262656L*x1) + (262656L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (787968L*x0) + (787968L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                auto tmp5 = c10::convert<long>(c10::div_floor_integer((x3 + (512L*x2)), 513L));
                                auto tmp6 = static_cast<long>(256);
                                auto tmp7 = tmp5 < tmp6;
                                auto tmp8 = [&]
                                {
                                    auto tmp9 = c10::convert<long>(static_cast<long>((x3 + (512L*x2))) % static_cast<long>(513L));
                                    auto tmp10 = static_cast<long>(257);
                                    auto tmp11 = tmp9 < tmp10;
                                    auto tmp12 = [&]
                                    {
                                        auto tmp13 = out_ptr12[static_cast<long>(256L + x3 + (512L*x2) + (131328L*x1) + (131328L*(c10::div_floor_integer((x3 + (512L*x2)), 262656L))) + (525312L*x0) + (525312L*(c10::div_floor_integer((x3 + (512L*x2) + (262656L*x1)), 787968L))))];
                                        return tmp13;
                                    }
                                    ;
                                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = tmp11 ? tmp14 : tmp15;
                                    return tmp16;
                                }
                                ;
                                auto tmp17 = tmp7 ? tmp8() : static_cast<decltype(tmp8())>(0.0);
                                auto tmp18 = static_cast<float>(0.0);
                                auto tmp19 = tmp7 ? tmp17 : tmp18;
                                auto tmp20 = decltype(tmp4)(tmp4 + tmp19);
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                            out_ptr14[static_cast<long>(x3 + (512L*x2) + (262144L*x1) + (786432L*x0))] = tmp21;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_index_add_127 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_add_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_as_strided_scatter_div_view_129 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp3.store(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                tmp0.store(out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = c10::convert<int>(256L + x0);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<int>(1536);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = tmp2 & tmp4;
                    auto tmp6 = [&]
                    {
                        auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(16384L + (64L*x0) + (98304L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return masked_load(tmpbuf, to_float_mask(tmp5)); })();
                        return tmp7;
                    }
                    ;
                    auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_130 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_sum_131 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_sum_132 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_9, primals_15, primals_25, primals_31, primals_41, primals_47, primals_57, primals_63, primals_73, primals_79, primals_89, primals_95, primals_105, primals_111, primals_121, primals_127, primals_137, primals_143, primals_153, primals_159, primals_169, primals_175, primals_185, primals_191, view, slice_64, rev_1, unsqueeze_16, getitem_1, view_69, getitem_3, mul_1, view_71, addmm_4, view_73, getitem_7, mul_6, view_75, getitem_11, view_144, getitem_13, mul_9, view_146, addmm_10, view_148, getitem_17, mul_14, view_150, getitem_21, view_219, getitem_23, mul_17, view_221, addmm_16, view_223, getitem_27, mul_22, view_225, getitem_31, view_294, getitem_33, mul_25, view_296, addmm_22, view_298, getitem_37, mul_30, view_300, getitem_41, view_369, getitem_43, mul_33, view_371, addmm_28, view_373, getitem_47, mul_38, view_375, getitem_51, view_444, getitem_53, mul_41, view_446, addmm_34, view_448, getitem_57, mul_46, view_450, getitem_61, view_519, getitem_63, mul_49, view_521, addmm_40, view_523, getitem_67, mul_54, view_525, getitem_71, view_594, getitem_73, mul_57, view_596, addmm_46, view_598, getitem_77, mul_62, view_600, getitem_81, view_669, getitem_83, mul_65, view_671, addmm_52, view_673, getitem_87, mul_70, view_675, getitem_91, view_744, getitem_93, mul_73, view_746, addmm_58, view_748, getitem_97, mul_78, view_750, getitem_101, view_819, getitem_103, mul_81, view_821, addmm_64, view_823, getitem_107, mul_86, view_825, getitem_111, view_894, getitem_113, mul_89, view_896, addmm_70, view_898, getitem_117, mul_94, div_120, permute_756, permute_760, div_121, permute_764, permute_772, permute_773, alias_12, permute_783, permute_784, permute_795, permute_799, permute_808, div_123, permute_814, permute_818, div_124, permute_822, permute_830, permute_831, alias_13, permute_841, permute_842, permute_853, permute_857, permute_866, div_126, permute_872, permute_876, div_127, permute_880, permute_888, permute_889, alias_14, permute_899, permute_900, permute_911, permute_915, permute_924, div_129, permute_930, permute_934, div_130, permute_938, permute_946, permute_947, alias_15, permute_957, permute_958, permute_969, permute_973, permute_982, div_132, permute_988, permute_992, div_133, permute_996, permute_1004, permute_1005, alias_16, permute_1015, permute_1016, permute_1027, permute_1031, permute_1040, div_135, permute_1046, permute_1050, div_136, permute_1054, permute_1062, permute_1063, alias_17, permute_1073, permute_1074, permute_1085, permute_1089, permute_1098, div_138, permute_1104, permute_1108, div_139, permute_1112, permute_1120, permute_1121, alias_18, permute_1131, permute_1132, permute_1143, permute_1147, permute_1156, div_141, permute_1162, permute_1166, div_142, permute_1170, permute_1178, permute_1179, alias_19, permute_1189, permute_1190, permute_1201, permute_1205, permute_1214, div_144, permute_1220, permute_1224, div_145, permute_1228, permute_1236, permute_1237, alias_20, permute_1247, permute_1248, permute_1259, permute_1263, permute_1272, div_147, permute_1278, permute_1282, div_148, permute_1286, permute_1294, permute_1295, alias_21, permute_1305, permute_1306, permute_1317, permute_1321, permute_1330, div_150, permute_1336, permute_1340, div_151, permute_1344, permute_1352, permute_1353, alias_22, permute_1363, permute_1364, permute_1375, permute_1379, permute_1388, div_153, permute_1394, permute_1398, div_154, permute_1402, permute_1410, permute_1411, alias_23, permute_1421, permute_1422, permute_1433, permute_1437, permute_1446, tangents_1 = args
    args.clear()
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_153, (768, ), (1, ))
    assert_size_stride(primals_159, (768, ), (1, ))
    assert_size_stride(primals_169, (768, ), (1, ))
    assert_size_stride(primals_175, (768, ), (1, ))
    assert_size_stride(primals_185, (768, ), (1, ))
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(view, (1024, 768), (768, 1))
    assert_size_stride(slice_64, (1, 256, 1, 257), (65792, 257, 257, 1))
    assert_size_stride(rev_1, (1, 256, 1, 257), (65792, 257, 257, 1))
    assert_size_stride(unsqueeze_16, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(getitem_1, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_69, (1024, 768), (768, 1))
    assert_size_stride(getitem_3, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_1, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_71, (1024, 768), (768, 1))
    assert_size_stride(addmm_4, (1024, 3072), (3072, 1))
    assert_size_stride(view_73, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_7, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_6, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_75, (1024, 768), (768, 1))
    assert_size_stride(getitem_11, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_144, (1024, 768), (768, 1))
    assert_size_stride(getitem_13, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_9, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_146, (1024, 768), (768, 1))
    assert_size_stride(addmm_10, (1024, 3072), (3072, 1))
    assert_size_stride(view_148, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_17, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_14, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_150, (1024, 768), (768, 1))
    assert_size_stride(getitem_21, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_219, (1024, 768), (768, 1))
    assert_size_stride(getitem_23, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_17, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_221, (1024, 768), (768, 1))
    assert_size_stride(addmm_16, (1024, 3072), (3072, 1))
    assert_size_stride(view_223, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_27, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_22, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_225, (1024, 768), (768, 1))
    assert_size_stride(getitem_31, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_294, (1024, 768), (768, 1))
    assert_size_stride(getitem_33, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_25, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_296, (1024, 768), (768, 1))
    assert_size_stride(addmm_22, (1024, 3072), (3072, 1))
    assert_size_stride(view_298, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_37, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_30, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_300, (1024, 768), (768, 1))
    assert_size_stride(getitem_41, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_369, (1024, 768), (768, 1))
    assert_size_stride(getitem_43, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_33, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_371, (1024, 768), (768, 1))
    assert_size_stride(addmm_28, (1024, 3072), (3072, 1))
    assert_size_stride(view_373, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_47, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_38, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_375, (1024, 768), (768, 1))
    assert_size_stride(getitem_51, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_444, (1024, 768), (768, 1))
    assert_size_stride(getitem_53, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_41, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_446, (1024, 768), (768, 1))
    assert_size_stride(addmm_34, (1024, 3072), (3072, 1))
    assert_size_stride(view_448, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_57, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_46, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_450, (1024, 768), (768, 1))
    assert_size_stride(getitem_61, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_519, (1024, 768), (768, 1))
    assert_size_stride(getitem_63, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_49, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_521, (1024, 768), (768, 1))
    assert_size_stride(addmm_40, (1024, 3072), (3072, 1))
    assert_size_stride(view_523, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_67, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_54, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_525, (1024, 768), (768, 1))
    assert_size_stride(getitem_71, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_594, (1024, 768), (768, 1))
    assert_size_stride(getitem_73, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_57, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_596, (1024, 768), (768, 1))
    assert_size_stride(addmm_46, (1024, 3072), (3072, 1))
    assert_size_stride(view_598, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_77, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_62, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_600, (1024, 768), (768, 1))
    assert_size_stride(getitem_81, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_669, (1024, 768), (768, 1))
    assert_size_stride(getitem_83, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_65, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_671, (1024, 768), (768, 1))
    assert_size_stride(addmm_52, (1024, 3072), (3072, 1))
    assert_size_stride(view_673, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_87, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_70, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_675, (1024, 768), (768, 1))
    assert_size_stride(getitem_91, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_744, (1024, 768), (768, 1))
    assert_size_stride(getitem_93, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_73, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_746, (1024, 768), (768, 1))
    assert_size_stride(addmm_58, (1024, 3072), (3072, 1))
    assert_size_stride(view_748, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_97, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_78, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_750, (1024, 768), (768, 1))
    assert_size_stride(getitem_101, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_819, (1024, 768), (768, 1))
    assert_size_stride(getitem_103, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_81, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_821, (1024, 768), (768, 1))
    assert_size_stride(addmm_64, (1024, 3072), (3072, 1))
    assert_size_stride(view_823, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_107, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_86, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_825, (1024, 768), (768, 1))
    assert_size_stride(getitem_111, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_894, (1024, 768), (768, 1))
    assert_size_stride(getitem_113, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_89, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_896, (1024, 768), (768, 1))
    assert_size_stride(addmm_70, (1024, 3072), (3072, 1))
    assert_size_stride(view_898, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_117, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_94, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(div_120, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_756, (768, 3072), (3072, 1))
    assert_size_stride(permute_760, (3072, 768), (768, 1))
    assert_size_stride(div_121, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_764, (768, 768), (768, 1))
    assert_size_stride(permute_772, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_773, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_12, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_783, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_784, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_795, (768, 768), (768, 1))
    assert_size_stride(permute_799, (768, 768), (768, 1))
    assert_size_stride(permute_808, (768, 768), (768, 1))
    assert_size_stride(div_123, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_814, (768, 3072), (3072, 1))
    assert_size_stride(permute_818, (3072, 768), (768, 1))
    assert_size_stride(div_124, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_822, (768, 768), (768, 1))
    assert_size_stride(permute_830, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_831, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_13, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_841, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_842, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_853, (768, 768), (768, 1))
    assert_size_stride(permute_857, (768, 768), (768, 1))
    assert_size_stride(permute_866, (768, 768), (768, 1))
    assert_size_stride(div_126, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_872, (768, 3072), (3072, 1))
    assert_size_stride(permute_876, (3072, 768), (768, 1))
    assert_size_stride(div_127, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_880, (768, 768), (768, 1))
    assert_size_stride(permute_888, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_889, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_14, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_899, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_900, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_911, (768, 768), (768, 1))
    assert_size_stride(permute_915, (768, 768), (768, 1))
    assert_size_stride(permute_924, (768, 768), (768, 1))
    assert_size_stride(div_129, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_930, (768, 3072), (3072, 1))
    assert_size_stride(permute_934, (3072, 768), (768, 1))
    assert_size_stride(div_130, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_938, (768, 768), (768, 1))
    assert_size_stride(permute_946, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_947, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_15, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_957, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_958, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_969, (768, 768), (768, 1))
    assert_size_stride(permute_973, (768, 768), (768, 1))
    assert_size_stride(permute_982, (768, 768), (768, 1))
    assert_size_stride(div_132, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_988, (768, 3072), (3072, 1))
    assert_size_stride(permute_992, (3072, 768), (768, 1))
    assert_size_stride(div_133, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_996, (768, 768), (768, 1))
    assert_size_stride(permute_1004, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_1005, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_16, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_1015, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_1016, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_1027, (768, 768), (768, 1))
    assert_size_stride(permute_1031, (768, 768), (768, 1))
    assert_size_stride(permute_1040, (768, 768), (768, 1))
    assert_size_stride(div_135, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1046, (768, 3072), (3072, 1))
    assert_size_stride(permute_1050, (3072, 768), (768, 1))
    assert_size_stride(div_136, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1054, (768, 768), (768, 1))
    assert_size_stride(permute_1062, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_1063, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_17, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_1073, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_1074, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_1085, (768, 768), (768, 1))
    assert_size_stride(permute_1089, (768, 768), (768, 1))
    assert_size_stride(permute_1098, (768, 768), (768, 1))
    assert_size_stride(div_138, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1104, (768, 3072), (3072, 1))
    assert_size_stride(permute_1108, (3072, 768), (768, 1))
    assert_size_stride(div_139, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1112, (768, 768), (768, 1))
    assert_size_stride(permute_1120, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_1121, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_18, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_1131, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_1132, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_1143, (768, 768), (768, 1))
    assert_size_stride(permute_1147, (768, 768), (768, 1))
    assert_size_stride(permute_1156, (768, 768), (768, 1))
    assert_size_stride(div_141, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1162, (768, 3072), (3072, 1))
    assert_size_stride(permute_1166, (3072, 768), (768, 1))
    assert_size_stride(div_142, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1170, (768, 768), (768, 1))
    assert_size_stride(permute_1178, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_1179, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_19, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_1189, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_1190, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_1201, (768, 768), (768, 1))
    assert_size_stride(permute_1205, (768, 768), (768, 1))
    assert_size_stride(permute_1214, (768, 768), (768, 1))
    assert_size_stride(div_144, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1220, (768, 3072), (3072, 1))
    assert_size_stride(permute_1224, (3072, 768), (768, 1))
    assert_size_stride(div_145, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1228, (768, 768), (768, 1))
    assert_size_stride(permute_1236, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_1237, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_20, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_1247, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_1248, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_1259, (768, 768), (768, 1))
    assert_size_stride(permute_1263, (768, 768), (768, 1))
    assert_size_stride(permute_1272, (768, 768), (768, 1))
    assert_size_stride(div_147, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1278, (768, 3072), (3072, 1))
    assert_size_stride(permute_1282, (3072, 768), (768, 1))
    assert_size_stride(div_148, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1286, (768, 768), (768, 1))
    assert_size_stride(permute_1294, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_1295, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_21, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_1305, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_1306, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_1317, (768, 768), (768, 1))
    assert_size_stride(permute_1321, (768, 768), (768, 1))
    assert_size_stride(permute_1330, (768, 768), (768, 1))
    assert_size_stride(div_150, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1336, (768, 3072), (3072, 1))
    assert_size_stride(permute_1340, (3072, 768), (768, 1))
    assert_size_stride(div_151, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1344, (768, 768), (768, 1))
    assert_size_stride(permute_1352, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_1353, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_22, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_1363, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_1364, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_1375, (768, 768), (768, 1))
    assert_size_stride(permute_1379, (768, 768), (768, 1))
    assert_size_stride(permute_1388, (768, 768), (768, 1))
    assert_size_stride(div_153, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1394, (768, 3072), (3072, 1))
    assert_size_stride(permute_1398, (3072, 768), (768, 1))
    assert_size_stride(div_154, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1402, (768, 768), (768, 1))
    assert_size_stride(permute_1410, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_1411, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_23, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_1421, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_1422, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_1433, (768, 768), (768, 1))
    assert_size_stride(permute_1437, (768, 768), (768, 1))
    assert_size_stride(permute_1446, (768, 768), (768, 1))
    assert_size_stride(tangents_1, (1, 1024, 768), (786432, 768, 1))
    buf0 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf2 = empty((1, 1024, 768), device='cpu', dtype=torch.float32)
    buf3 = empty((768, ), device='cpu', dtype=torch.float32)
    buf4 = empty((768, ), device='cpu', dtype=torch.float32)
    buf5 = empty((1, 1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_native_dropout_backward_native_layer_norm_backward_0(c_void_p(tangents_1.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(mul_94.data_ptr()), c_void_p(div_120.data_ptr()), c_void_p(getitem_117.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()))
    del div_120
    del getitem_117
    del mul_94
    del primals_191
    del tangents_1
    buf6 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (1024, 768), (768, 1), 0), permute_756, out=buf6)
    del permute_756
    buf7 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (768, 1024), (1, 768), 0), view_898, out=buf7)
    del view_898
    buf8 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf9 = reinterpret_tensor(buf6, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf6  # reuse
    cpp_fused_gelu_gelu_backward_sum_1(c_void_p(buf9.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(addmm_70.data_ptr()), c_void_p(buf8.data_ptr()))
    del addmm_70
    buf10 = reinterpret_tensor(buf5, (1024, 768), (768, 1), 0); del buf5  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf9, (1024, 3072), (3072, 1), 0), permute_760, out=buf10)
    del permute_760
    buf11 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf9, (3072, 1024), (1, 3072), 0), view_896, out=buf11)
    del view_896
    buf12 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf13 = buf1; del buf1  # reuse
    buf14 = buf0; del buf0  # reuse
    buf15 = empty((1, 1024, 768), device='cpu', dtype=torch.float32)
    buf16 = empty((768, ), device='cpu', dtype=torch.float32)
    buf17 = empty((768, ), device='cpu', dtype=torch.float32)
    buf18 = empty((1, 1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_2(c_void_p(buf9.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(mul_89.data_ptr()), c_void_p(div_121.data_ptr()), c_void_p(getitem_113.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()))
    del div_121
    del getitem_113
    del mul_89
    del primals_185
    buf19 = reinterpret_tensor(buf2, (1024, 768), (768, 1), 0); del buf2  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf18, (1024, 768), (768, 1), 0), permute_764, out=buf19)
    del permute_764
    buf20 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf18, (768, 1024), (1, 768), 0), view_894, out=buf20)
    del view_894
    buf21 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf22 = reinterpret_tensor(buf10, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf10  # reuse
    cpp_fused_clone_sum_3(c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()))
    buf23 = empty((48, 768, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_772, reinterpret_tensor(buf22, (48, 256, 64), (16384, 64, 1), 0), out=buf23)
    del permute_772
    buf24 = empty((48, 256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf22, (48, 256, 64), (16384, 64, 1), 0), permute_773, out=buf24)
    del permute_773
    buf25 = empty((1179648, ), device='cpu', dtype=torch.int64)
    buf26 = empty((1179648, ), device='cpu', dtype=torch.float32)
    buf27 = empty((12, 4, 768, 64), device='cpu', dtype=torch.int64)
    cpp_fused_arange_clone_index_add_new_zeros_4(c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()))
    aten.index_put_(buf26, [reinterpret_tensor(buf27, (2359296, ), (1, ), 0)], reinterpret_tensor(buf23, (2359296, ), (1, ), 0), True)
    buf30 = empty_strided((1, 1024, 12, 1), (12288, 12, 1, 12288), device='cpu', dtype=torch.float32)
    buf32 = empty_strided((1024, 12, 513), (513, 525312, 1), device='cpu', dtype=torch.float32)
    buf33 = empty((6303744, ), device='cpu', dtype=torch.float32)
    buf36 = empty_strided((1, 1024, 12, 513), (6303744, 513, 525312, 1), device='cpu', dtype=torch.float32)
    buf37 = empty((12, 4, 256, 513), device='cpu', dtype=torch.float32)
    buf40 = empty((12, 4, 256, 513), device='cpu', dtype=torch.float32)
    buf42 = empty_strided((1024, 12, 513), (513, 525312, 1), device='cpu', dtype=torch.float32)
    buf43 = empty((6303744, ), device='cpu', dtype=torch.float32)
    buf45 = empty((12, 4, 256, 513), device='cpu', dtype=torch.float32)
    buf47 = empty((12, 4, 256, 513), device='cpu', dtype=torch.float32)
    buf49 = buf45; del buf45  # reuse
    buf51 = empty((12, 4, 256, 513), device='cpu', dtype=torch.float32)
    buf54 = empty((12, 4, 256, 513), device='cpu', dtype=torch.float32)
    buf57 = empty((12, 4, 256, 513), device='cpu', dtype=torch.float32)
    buf59 = empty((12, 3, 512, 513), device='cpu', dtype=torch.float32)
    buf61 = empty((12, 3, 512, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_5(c_void_p(buf49.data_ptr()), c_void_p(unsqueeze_16.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(getitem_111.data_ptr()), c_void_p(alias_12.data_ptr()), c_void_p(rev_1.data_ptr()), c_void_p(slice_64.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf61.data_ptr()))
    del alias_12
    del getitem_111
    buf62 = empty((36, 64, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_783, reinterpret_tensor(buf61, (36, 512, 512), (262144, 512, 1), 0), out=buf62)
    del permute_783
    buf63 = empty((36, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf61, (36, 512, 512), (262144, 512, 1), 0), permute_784, out=buf63)
    del permute_784
    buf65 = empty((786432, ), device='cpu', dtype=torch.int64)
    buf66 = reinterpret_tensor(buf22, (786432, ), (1, ), 0); del buf22  # reuse
    buf67 = empty((12, 3, 512, 64), device='cpu', dtype=torch.float32)
    buf68 = reinterpret_tensor(buf25, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf25  # reuse
    cpp_fused_arange_clone_index_add_6(c_void_p(buf62.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()))
    del buf65
    aten.index_put_(buf66, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf67, (1179648, ), (1, ), 0), True)
    buf71 = reinterpret_tensor(buf19, (786432, ), (1, ), 0); del buf19  # reuse
    cpp_fused_index_add_7(c_void_p(buf71.data_ptr()))
    aten.index_put_(buf71, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf63, (1179648, ), (1, ), 0), True)
    buf74 = reinterpret_tensor(buf18, (786432, ), (1, ), 0); del buf18  # reuse
    buf77 = empty((1024, 1, 768), device='cpu', dtype=torch.float32)
    buf78 = empty((1024, 768), device='cpu', dtype=torch.float32)
    buf80 = empty((1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_as_strided_scatter_div_view_8(c_void_p(buf71.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf80.data_ptr()))
    buf81 = reinterpret_tensor(buf77, (1024, 768), (768, 1), 0); del buf77  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf80, permute_795, out=buf81)
    del permute_795
    buf82 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf80, (768, 1024), (1, 768), 0), view_825, out=buf82)
    buf83 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_9(c_void_p(buf80.data_ptr()), c_void_p(buf83.data_ptr()))
    buf84 = buf80; del buf80  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf66, (1024, 768), (768, 1), 0), permute_799, out=buf84)
    del permute_799
    buf85 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf66, (768, 1024), (1, 768), 0), view_825, out=buf85)
    buf86 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_10(c_void_p(buf66.data_ptr()), c_void_p(buf86.data_ptr()))
    buf87 = reinterpret_tensor(buf66, (1024, 768), (768, 1), 0); del buf66  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf78, permute_808, out=buf87)
    del permute_808
    buf88 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf78, (768, 1024), (1, 768), 0), view_825, out=buf88)
    del view_825
    buf89 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf90 = reinterpret_tensor(buf74, (1, 1024, 768), (786432, 768, 1), 0); del buf74  # reuse
    buf91 = buf14; del buf14  # reuse
    buf92 = buf13; del buf13  # reuse
    buf93 = buf90; del buf90  # reuse
    buf94 = empty((768, ), device='cpu', dtype=torch.float32)
    buf95 = empty((768, ), device='cpu', dtype=torch.float32)
    buf96 = reinterpret_tensor(buf71, (1, 1024, 768), (786432, 768, 1), 0); del buf71  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_11(c_void_p(buf93.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(mul_86.data_ptr()), c_void_p(div_123.data_ptr()), c_void_p(getitem_107.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()))
    del div_123
    del getitem_107
    del mul_86
    del primals_175
    buf97 = reinterpret_tensor(buf9, (1024, 3072), (3072, 1), 0); del buf9  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf96, (1024, 768), (768, 1), 0), permute_814, out=buf97)
    del permute_814
    buf98 = reinterpret_tensor(buf23, (768, 3072), (3072, 1), 0); del buf23  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf96, (768, 1024), (1, 768), 0), view_823, out=buf98)
    del view_823
    buf99 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf100 = reinterpret_tensor(buf97, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf97  # reuse
    cpp_fused_gelu_gelu_backward_sum_12(c_void_p(buf100.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(addmm_64.data_ptr()), c_void_p(buf99.data_ptr()))
    del addmm_64
    buf101 = reinterpret_tensor(buf96, (1024, 768), (768, 1), 0); del buf96  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf100, (1024, 3072), (3072, 1), 0), permute_818, out=buf101)
    del permute_818
    buf102 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf100, (3072, 1024), (1, 3072), 0), view_821, out=buf102)
    del view_821
    buf103 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf104 = buf92; del buf92  # reuse
    buf105 = buf91; del buf91  # reuse
    buf106 = reinterpret_tensor(buf87, (1, 1024, 768), (786432, 768, 1), 0); del buf87  # reuse
    buf107 = empty((768, ), device='cpu', dtype=torch.float32)
    buf108 = empty((768, ), device='cpu', dtype=torch.float32)
    buf109 = reinterpret_tensor(buf84, (1, 1024, 768), (786432, 768, 1), 0); del buf84  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_13(c_void_p(buf100.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(mul_81.data_ptr()), c_void_p(div_124.data_ptr()), c_void_p(getitem_103.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    del div_124
    del getitem_103
    del mul_81
    del primals_169
    buf110 = reinterpret_tensor(buf93, (1024, 768), (768, 1), 0); del buf93  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf109, (1024, 768), (768, 1), 0), permute_822, out=buf110)
    del permute_822
    buf111 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf109, (768, 1024), (1, 768), 0), view_819, out=buf111)
    del view_819
    buf112 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf113 = reinterpret_tensor(buf101, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf101  # reuse
    cpp_fused_clone_sum_14(c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()))
    buf114 = empty((48, 768, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_830, reinterpret_tensor(buf113, (48, 256, 64), (16384, 64, 1), 0), out=buf114)
    del permute_830
    buf115 = reinterpret_tensor(buf61, (48, 256, 768), (196608, 768, 1), 0); del buf61  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf113, (48, 256, 64), (16384, 64, 1), 0), permute_831, out=buf115)
    del permute_831
    buf116 = buf26; del buf26  # reuse
    cpp_fused_index_add_new_zeros_15(c_void_p(buf116.data_ptr()))
    aten.index_put_(buf116, [reinterpret_tensor(buf27, (2359296, ), (1, ), 0)], reinterpret_tensor(buf114, (2359296, ), (1, ), 0), True)
    buf119 = buf30; del buf30  # reuse
    buf120 = reinterpret_tensor(buf57, (1024, 12, 513), (513, 525312, 1), 0); del buf57  # reuse
    buf121 = reinterpret_tensor(buf54, (6303744, ), (1, ), 0); del buf54  # reuse
    buf124 = reinterpret_tensor(buf51, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf51  # reuse
    buf125 = buf49; del buf49  # reuse
    buf128 = buf47; del buf47  # reuse
    buf130 = reinterpret_tensor(buf43, (1024, 12, 513), (513, 525312, 1), 0); del buf43  # reuse
    buf131 = reinterpret_tensor(buf42, (6303744, ), (1, ), 0); del buf42  # reuse
    buf133 = buf40; del buf40  # reuse
    buf135 = buf37; del buf37  # reuse
    buf137 = buf133; del buf133  # reuse
    buf139 = reinterpret_tensor(buf36, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf36  # reuse
    buf142 = reinterpret_tensor(buf33, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf33  # reuse
    buf145 = reinterpret_tensor(buf32, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf32  # reuse
    buf147 = buf59; del buf59  # reuse
    buf149 = reinterpret_tensor(buf24, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf24  # reuse
    cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_16(c_void_p(buf137.data_ptr()), c_void_p(unsqueeze_16.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(getitem_101.data_ptr()), c_void_p(alias_13.data_ptr()), c_void_p(rev_1.data_ptr()), c_void_p(slice_64.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf149.data_ptr()))
    del alias_13
    del getitem_101
    buf150 = reinterpret_tensor(buf63, (36, 64, 512), (32768, 512, 1), 0); del buf63  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_841, reinterpret_tensor(buf149, (36, 512, 512), (262144, 512, 1), 0), out=buf150)
    del permute_841
    buf151 = reinterpret_tensor(buf67, (36, 512, 64), (32768, 64, 1), 0); del buf67  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf149, (36, 512, 512), (262144, 512, 1), 0), permute_842, out=buf151)
    del permute_842
    buf152 = reinterpret_tensor(buf113, (786432, ), (1, ), 0); del buf113  # reuse
    buf153 = reinterpret_tensor(buf62, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf62  # reuse
    cpp_fused_clone_index_add_17(c_void_p(buf150.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()))
    aten.index_put_(buf152, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf153, (1179648, ), (1, ), 0), True)
    buf156 = reinterpret_tensor(buf110, (786432, ), (1, ), 0); del buf110  # reuse
    cpp_fused_index_add_18(c_void_p(buf156.data_ptr()))
    aten.index_put_(buf156, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf151, (1179648, ), (1, ), 0), True)
    buf159 = reinterpret_tensor(buf109, (786432, ), (1, ), 0); del buf109  # reuse
    buf162 = reinterpret_tensor(buf81, (1024, 1, 768), (768, 768, 1), 0); del buf81  # reuse
    buf163 = buf78; del buf78  # reuse
    buf165 = reinterpret_tensor(buf15, (1024, 768), (768, 1), 0); del buf15  # reuse
    cpp_fused_as_strided_scatter_div_view_19(c_void_p(buf156.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf165.data_ptr()))
    buf166 = reinterpret_tensor(buf162, (1024, 768), (768, 1), 0); del buf162  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf165, permute_853, out=buf166)
    del permute_853
    buf167 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf165, (768, 1024), (1, 768), 0), view_750, out=buf167)
    buf168 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_20(c_void_p(buf165.data_ptr()), c_void_p(buf168.data_ptr()))
    buf169 = buf165; del buf165  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf152, (1024, 768), (768, 1), 0), permute_857, out=buf169)
    del permute_857
    buf170 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf152, (768, 1024), (1, 768), 0), view_750, out=buf170)
    buf171 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_21(c_void_p(buf152.data_ptr()), c_void_p(buf171.data_ptr()))
    buf172 = reinterpret_tensor(buf152, (1024, 768), (768, 1), 0); del buf152  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf163, permute_866, out=buf172)
    del permute_866
    buf173 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf163, (768, 1024), (1, 768), 0), view_750, out=buf173)
    del view_750
    buf174 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf175 = reinterpret_tensor(buf159, (1, 1024, 768), (786432, 768, 1), 0); del buf159  # reuse
    buf176 = buf105; del buf105  # reuse
    buf177 = buf104; del buf104  # reuse
    buf178 = buf175; del buf175  # reuse
    buf179 = empty((768, ), device='cpu', dtype=torch.float32)
    buf180 = empty((768, ), device='cpu', dtype=torch.float32)
    buf181 = reinterpret_tensor(buf156, (1, 1024, 768), (786432, 768, 1), 0); del buf156  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_22(c_void_p(buf178.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(mul_78.data_ptr()), c_void_p(div_126.data_ptr()), c_void_p(getitem_97.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()))
    del div_126
    del getitem_97
    del mul_78
    del primals_159
    buf182 = reinterpret_tensor(buf100, (1024, 3072), (3072, 1), 0); del buf100  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf181, (1024, 768), (768, 1), 0), permute_872, out=buf182)
    del permute_872
    buf183 = reinterpret_tensor(buf114, (768, 3072), (3072, 1), 0); del buf114  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf181, (768, 1024), (1, 768), 0), view_748, out=buf183)
    del view_748
    buf184 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf185 = reinterpret_tensor(buf182, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf182  # reuse
    cpp_fused_gelu_gelu_backward_sum_23(c_void_p(buf185.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(addmm_58.data_ptr()), c_void_p(buf184.data_ptr()))
    del addmm_58
    buf186 = reinterpret_tensor(buf181, (1024, 768), (768, 1), 0); del buf181  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf185, (1024, 3072), (3072, 1), 0), permute_876, out=buf186)
    del permute_876
    buf187 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf185, (3072, 1024), (1, 3072), 0), view_746, out=buf187)
    del view_746
    buf188 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf189 = buf177; del buf177  # reuse
    buf190 = buf176; del buf176  # reuse
    buf191 = reinterpret_tensor(buf172, (1, 1024, 768), (786432, 768, 1), 0); del buf172  # reuse
    buf192 = empty((768, ), device='cpu', dtype=torch.float32)
    buf193 = empty((768, ), device='cpu', dtype=torch.float32)
    buf194 = reinterpret_tensor(buf169, (1, 1024, 768), (786432, 768, 1), 0); del buf169  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_24(c_void_p(buf185.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(mul_73.data_ptr()), c_void_p(div_127.data_ptr()), c_void_p(getitem_93.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()))
    del div_127
    del getitem_93
    del mul_73
    del primals_153
    buf195 = buf186; del buf186  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf194, (1024, 768), (768, 1), 0), permute_880, out=buf195)
    del permute_880
    buf196 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf194, (768, 1024), (1, 768), 0), view_744, out=buf196)
    del view_744
    buf197 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf198 = reinterpret_tensor(buf178, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf178  # reuse
    cpp_fused_clone_sum_25(c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()))
    buf199 = empty((48, 768, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_888, reinterpret_tensor(buf198, (48, 256, 64), (16384, 64, 1), 0), out=buf199)
    del permute_888
    buf200 = reinterpret_tensor(buf149, (48, 256, 768), (196608, 768, 1), 0); del buf149  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf198, (48, 256, 64), (16384, 64, 1), 0), permute_889, out=buf200)
    del permute_889
    buf201 = buf116; del buf116  # reuse
    cpp_fused_index_add_new_zeros_26(c_void_p(buf201.data_ptr()))
    aten.index_put_(buf201, [reinterpret_tensor(buf27, (2359296, ), (1, ), 0)], reinterpret_tensor(buf199, (2359296, ), (1, ), 0), True)
    buf204 = buf119; del buf119  # reuse
    buf205 = reinterpret_tensor(buf145, (1024, 12, 513), (513, 525312, 1), 0); del buf145  # reuse
    buf206 = reinterpret_tensor(buf142, (6303744, ), (1, ), 0); del buf142  # reuse
    buf209 = reinterpret_tensor(buf139, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf139  # reuse
    buf210 = buf137; del buf137  # reuse
    buf213 = buf135; del buf135  # reuse
    buf215 = reinterpret_tensor(buf131, (1024, 12, 513), (513, 525312, 1), 0); del buf131  # reuse
    buf216 = reinterpret_tensor(buf130, (6303744, ), (1, ), 0); del buf130  # reuse
    buf218 = buf128; del buf128  # reuse
    buf220 = buf125; del buf125  # reuse
    buf222 = buf218; del buf218  # reuse
    buf224 = reinterpret_tensor(buf124, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf124  # reuse
    buf227 = reinterpret_tensor(buf121, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf121  # reuse
    buf230 = reinterpret_tensor(buf120, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf120  # reuse
    buf232 = buf147; del buf147  # reuse
    buf234 = reinterpret_tensor(buf115, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf115  # reuse
    cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_27(c_void_p(buf222.data_ptr()), c_void_p(unsqueeze_16.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(getitem_91.data_ptr()), c_void_p(alias_14.data_ptr()), c_void_p(rev_1.data_ptr()), c_void_p(slice_64.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf234.data_ptr()))
    del alias_14
    del getitem_91
    buf235 = reinterpret_tensor(buf151, (36, 64, 512), (32768, 512, 1), 0); del buf151  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_899, reinterpret_tensor(buf234, (36, 512, 512), (262144, 512, 1), 0), out=buf235)
    del permute_899
    buf236 = reinterpret_tensor(buf153, (36, 512, 64), (32768, 64, 1), 0); del buf153  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf234, (36, 512, 512), (262144, 512, 1), 0), permute_900, out=buf236)
    del permute_900
    buf237 = reinterpret_tensor(buf198, (786432, ), (1, ), 0); del buf198  # reuse
    buf238 = reinterpret_tensor(buf150, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf150  # reuse
    cpp_fused_clone_index_add_28(c_void_p(buf235.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()))
    aten.index_put_(buf237, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf238, (1179648, ), (1, ), 0), True)
    buf241 = reinterpret_tensor(buf195, (786432, ), (1, ), 0); del buf195  # reuse
    cpp_fused_index_add_29(c_void_p(buf241.data_ptr()))
    aten.index_put_(buf241, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf236, (1179648, ), (1, ), 0), True)
    buf244 = reinterpret_tensor(buf194, (786432, ), (1, ), 0); del buf194  # reuse
    buf247 = reinterpret_tensor(buf166, (1024, 1, 768), (768, 768, 1), 0); del buf166  # reuse
    buf248 = buf163; del buf163  # reuse
    buf250 = reinterpret_tensor(buf106, (1024, 768), (768, 1), 0); del buf106  # reuse
    cpp_fused_as_strided_scatter_div_view_30(c_void_p(buf241.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf250.data_ptr()))
    buf251 = reinterpret_tensor(buf247, (1024, 768), (768, 1), 0); del buf247  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf250, permute_911, out=buf251)
    del permute_911
    buf252 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf250, (768, 1024), (1, 768), 0), view_675, out=buf252)
    buf253 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_31(c_void_p(buf250.data_ptr()), c_void_p(buf253.data_ptr()))
    buf254 = buf250; del buf250  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf237, (1024, 768), (768, 1), 0), permute_915, out=buf254)
    del permute_915
    buf255 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf237, (768, 1024), (1, 768), 0), view_675, out=buf255)
    buf256 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_32(c_void_p(buf237.data_ptr()), c_void_p(buf256.data_ptr()))
    buf257 = reinterpret_tensor(buf237, (1024, 768), (768, 1), 0); del buf237  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf248, permute_924, out=buf257)
    del permute_924
    buf258 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf248, (768, 1024), (1, 768), 0), view_675, out=buf258)
    del view_675
    buf259 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf260 = reinterpret_tensor(buf244, (1, 1024, 768), (786432, 768, 1), 0); del buf244  # reuse
    buf261 = buf190; del buf190  # reuse
    buf262 = buf189; del buf189  # reuse
    buf263 = buf260; del buf260  # reuse
    buf264 = empty((768, ), device='cpu', dtype=torch.float32)
    buf265 = empty((768, ), device='cpu', dtype=torch.float32)
    buf266 = reinterpret_tensor(buf241, (1, 1024, 768), (786432, 768, 1), 0); del buf241  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_33(c_void_p(buf263.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(mul_70.data_ptr()), c_void_p(div_129.data_ptr()), c_void_p(getitem_87.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()))
    del div_129
    del getitem_87
    del mul_70
    del primals_143
    buf267 = reinterpret_tensor(buf185, (1024, 3072), (3072, 1), 0); del buf185  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf266, (1024, 768), (768, 1), 0), permute_930, out=buf267)
    del permute_930
    buf268 = reinterpret_tensor(buf199, (768, 3072), (3072, 1), 0); del buf199  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf266, (768, 1024), (1, 768), 0), view_673, out=buf268)
    del view_673
    buf269 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf270 = reinterpret_tensor(buf267, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf267  # reuse
    cpp_fused_gelu_gelu_backward_sum_34(c_void_p(buf270.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(addmm_52.data_ptr()), c_void_p(buf269.data_ptr()))
    del addmm_52
    buf271 = reinterpret_tensor(buf266, (1024, 768), (768, 1), 0); del buf266  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf270, (1024, 3072), (3072, 1), 0), permute_934, out=buf271)
    del permute_934
    buf272 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf270, (3072, 1024), (1, 3072), 0), view_671, out=buf272)
    del view_671
    buf273 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf274 = buf262; del buf262  # reuse
    buf275 = buf261; del buf261  # reuse
    buf276 = reinterpret_tensor(buf257, (1, 1024, 768), (786432, 768, 1), 0); del buf257  # reuse
    buf277 = empty((768, ), device='cpu', dtype=torch.float32)
    buf278 = empty((768, ), device='cpu', dtype=torch.float32)
    buf279 = reinterpret_tensor(buf254, (1, 1024, 768), (786432, 768, 1), 0); del buf254  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_35(c_void_p(buf270.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(mul_65.data_ptr()), c_void_p(div_130.data_ptr()), c_void_p(getitem_83.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()))
    del div_130
    del getitem_83
    del mul_65
    del primals_137
    buf280 = buf271; del buf271  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf279, (1024, 768), (768, 1), 0), permute_938, out=buf280)
    del permute_938
    buf281 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf279, (768, 1024), (1, 768), 0), view_669, out=buf281)
    del view_669
    buf282 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf283 = reinterpret_tensor(buf263, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf263  # reuse
    cpp_fused_clone_sum_36(c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()))
    buf284 = empty((48, 768, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_946, reinterpret_tensor(buf283, (48, 256, 64), (16384, 64, 1), 0), out=buf284)
    del permute_946
    buf285 = reinterpret_tensor(buf234, (48, 256, 768), (196608, 768, 1), 0); del buf234  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf283, (48, 256, 64), (16384, 64, 1), 0), permute_947, out=buf285)
    del permute_947
    buf286 = buf201; del buf201  # reuse
    cpp_fused_index_add_new_zeros_37(c_void_p(buf286.data_ptr()))
    aten.index_put_(buf286, [reinterpret_tensor(buf27, (2359296, ), (1, ), 0)], reinterpret_tensor(buf284, (2359296, ), (1, ), 0), True)
    buf289 = buf204; del buf204  # reuse
    buf290 = reinterpret_tensor(buf230, (1024, 12, 513), (513, 525312, 1), 0); del buf230  # reuse
    buf291 = reinterpret_tensor(buf227, (6303744, ), (1, ), 0); del buf227  # reuse
    buf294 = reinterpret_tensor(buf224, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf224  # reuse
    buf295 = buf222; del buf222  # reuse
    buf298 = buf220; del buf220  # reuse
    buf300 = reinterpret_tensor(buf216, (1024, 12, 513), (513, 525312, 1), 0); del buf216  # reuse
    buf301 = reinterpret_tensor(buf215, (6303744, ), (1, ), 0); del buf215  # reuse
    buf303 = buf213; del buf213  # reuse
    buf305 = buf210; del buf210  # reuse
    buf307 = buf303; del buf303  # reuse
    buf309 = reinterpret_tensor(buf209, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf209  # reuse
    buf312 = reinterpret_tensor(buf206, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf206  # reuse
    buf315 = reinterpret_tensor(buf205, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf205  # reuse
    buf317 = buf232; del buf232  # reuse
    buf319 = reinterpret_tensor(buf200, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf200  # reuse
    cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_38(c_void_p(buf307.data_ptr()), c_void_p(unsqueeze_16.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(getitem_81.data_ptr()), c_void_p(alias_15.data_ptr()), c_void_p(rev_1.data_ptr()), c_void_p(slice_64.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf319.data_ptr()))
    del alias_15
    del getitem_81
    buf320 = reinterpret_tensor(buf236, (36, 64, 512), (32768, 512, 1), 0); del buf236  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_957, reinterpret_tensor(buf319, (36, 512, 512), (262144, 512, 1), 0), out=buf320)
    del permute_957
    buf321 = reinterpret_tensor(buf238, (36, 512, 64), (32768, 64, 1), 0); del buf238  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf319, (36, 512, 512), (262144, 512, 1), 0), permute_958, out=buf321)
    del permute_958
    buf322 = reinterpret_tensor(buf283, (786432, ), (1, ), 0); del buf283  # reuse
    buf323 = reinterpret_tensor(buf235, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf235  # reuse
    cpp_fused_clone_index_add_39(c_void_p(buf320.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()))
    aten.index_put_(buf322, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf323, (1179648, ), (1, ), 0), True)
    buf326 = reinterpret_tensor(buf280, (786432, ), (1, ), 0); del buf280  # reuse
    cpp_fused_index_add_40(c_void_p(buf326.data_ptr()))
    aten.index_put_(buf326, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf321, (1179648, ), (1, ), 0), True)
    buf329 = reinterpret_tensor(buf279, (786432, ), (1, ), 0); del buf279  # reuse
    buf332 = reinterpret_tensor(buf251, (1024, 1, 768), (768, 768, 1), 0); del buf251  # reuse
    buf333 = buf248; del buf248  # reuse
    buf335 = reinterpret_tensor(buf191, (1024, 768), (768, 1), 0); del buf191  # reuse
    cpp_fused_as_strided_scatter_div_view_41(c_void_p(buf326.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf335.data_ptr()))
    buf336 = reinterpret_tensor(buf332, (1024, 768), (768, 1), 0); del buf332  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf335, permute_969, out=buf336)
    del permute_969
    buf337 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf335, (768, 1024), (1, 768), 0), view_600, out=buf337)
    buf338 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_42(c_void_p(buf335.data_ptr()), c_void_p(buf338.data_ptr()))
    buf339 = buf335; del buf335  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf322, (1024, 768), (768, 1), 0), permute_973, out=buf339)
    del permute_973
    buf340 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf322, (768, 1024), (1, 768), 0), view_600, out=buf340)
    buf341 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_43(c_void_p(buf322.data_ptr()), c_void_p(buf341.data_ptr()))
    buf342 = reinterpret_tensor(buf322, (1024, 768), (768, 1), 0); del buf322  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf333, permute_982, out=buf342)
    del permute_982
    buf343 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf333, (768, 1024), (1, 768), 0), view_600, out=buf343)
    del view_600
    buf344 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf345 = reinterpret_tensor(buf329, (1, 1024, 768), (786432, 768, 1), 0); del buf329  # reuse
    buf346 = buf275; del buf275  # reuse
    buf347 = buf274; del buf274  # reuse
    buf348 = buf345; del buf345  # reuse
    buf349 = empty((768, ), device='cpu', dtype=torch.float32)
    buf350 = empty((768, ), device='cpu', dtype=torch.float32)
    buf351 = reinterpret_tensor(buf326, (1, 1024, 768), (786432, 768, 1), 0); del buf326  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_44(c_void_p(buf348.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(mul_62.data_ptr()), c_void_p(div_132.data_ptr()), c_void_p(getitem_77.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()))
    del div_132
    del getitem_77
    del mul_62
    del primals_127
    buf352 = reinterpret_tensor(buf270, (1024, 3072), (3072, 1), 0); del buf270  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf351, (1024, 768), (768, 1), 0), permute_988, out=buf352)
    del permute_988
    buf353 = reinterpret_tensor(buf284, (768, 3072), (3072, 1), 0); del buf284  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf351, (768, 1024), (1, 768), 0), view_598, out=buf353)
    del view_598
    buf354 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf355 = reinterpret_tensor(buf352, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf352  # reuse
    cpp_fused_gelu_gelu_backward_sum_45(c_void_p(buf355.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(addmm_46.data_ptr()), c_void_p(buf354.data_ptr()))
    del addmm_46
    buf356 = reinterpret_tensor(buf351, (1024, 768), (768, 1), 0); del buf351  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf355, (1024, 3072), (3072, 1), 0), permute_992, out=buf356)
    del permute_992
    buf357 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf355, (3072, 1024), (1, 3072), 0), view_596, out=buf357)
    del view_596
    buf358 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf359 = buf347; del buf347  # reuse
    buf360 = buf346; del buf346  # reuse
    buf361 = reinterpret_tensor(buf342, (1, 1024, 768), (786432, 768, 1), 0); del buf342  # reuse
    buf362 = empty((768, ), device='cpu', dtype=torch.float32)
    buf363 = empty((768, ), device='cpu', dtype=torch.float32)
    buf364 = reinterpret_tensor(buf339, (1, 1024, 768), (786432, 768, 1), 0); del buf339  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_46(c_void_p(buf355.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(mul_57.data_ptr()), c_void_p(div_133.data_ptr()), c_void_p(getitem_73.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()))
    del div_133
    del getitem_73
    del mul_57
    del primals_121
    buf365 = buf356; del buf356  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf364, (1024, 768), (768, 1), 0), permute_996, out=buf365)
    del permute_996
    buf366 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf364, (768, 1024), (1, 768), 0), view_594, out=buf366)
    del view_594
    buf367 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf368 = reinterpret_tensor(buf348, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf348  # reuse
    cpp_fused_clone_sum_47(c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()))
    buf369 = empty((48, 768, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1004, reinterpret_tensor(buf368, (48, 256, 64), (16384, 64, 1), 0), out=buf369)
    del permute_1004
    buf370 = reinterpret_tensor(buf319, (48, 256, 768), (196608, 768, 1), 0); del buf319  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf368, (48, 256, 64), (16384, 64, 1), 0), permute_1005, out=buf370)
    del permute_1005
    buf371 = buf286; del buf286  # reuse
    cpp_fused_index_add_new_zeros_48(c_void_p(buf371.data_ptr()))
    aten.index_put_(buf371, [reinterpret_tensor(buf27, (2359296, ), (1, ), 0)], reinterpret_tensor(buf369, (2359296, ), (1, ), 0), True)
    buf374 = buf289; del buf289  # reuse
    buf375 = reinterpret_tensor(buf315, (1024, 12, 513), (513, 525312, 1), 0); del buf315  # reuse
    buf376 = reinterpret_tensor(buf312, (6303744, ), (1, ), 0); del buf312  # reuse
    buf379 = reinterpret_tensor(buf309, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf309  # reuse
    buf380 = buf307; del buf307  # reuse
    buf383 = buf305; del buf305  # reuse
    buf385 = reinterpret_tensor(buf301, (1024, 12, 513), (513, 525312, 1), 0); del buf301  # reuse
    buf386 = reinterpret_tensor(buf300, (6303744, ), (1, ), 0); del buf300  # reuse
    buf388 = buf298; del buf298  # reuse
    buf390 = buf295; del buf295  # reuse
    buf392 = buf388; del buf388  # reuse
    buf394 = reinterpret_tensor(buf294, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf294  # reuse
    buf397 = reinterpret_tensor(buf291, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf291  # reuse
    buf400 = reinterpret_tensor(buf290, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf290  # reuse
    buf402 = buf317; del buf317  # reuse
    buf404 = reinterpret_tensor(buf285, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf285  # reuse
    cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_49(c_void_p(buf392.data_ptr()), c_void_p(unsqueeze_16.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(getitem_71.data_ptr()), c_void_p(alias_16.data_ptr()), c_void_p(rev_1.data_ptr()), c_void_p(slice_64.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf404.data_ptr()))
    del alias_16
    del getitem_71
    buf405 = reinterpret_tensor(buf321, (36, 64, 512), (32768, 512, 1), 0); del buf321  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1015, reinterpret_tensor(buf404, (36, 512, 512), (262144, 512, 1), 0), out=buf405)
    del permute_1015
    buf406 = reinterpret_tensor(buf323, (36, 512, 64), (32768, 64, 1), 0); del buf323  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf404, (36, 512, 512), (262144, 512, 1), 0), permute_1016, out=buf406)
    del permute_1016
    buf407 = reinterpret_tensor(buf368, (786432, ), (1, ), 0); del buf368  # reuse
    buf408 = reinterpret_tensor(buf320, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf320  # reuse
    cpp_fused_clone_index_add_50(c_void_p(buf405.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf408.data_ptr()))
    aten.index_put_(buf407, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf408, (1179648, ), (1, ), 0), True)
    buf411 = reinterpret_tensor(buf365, (786432, ), (1, ), 0); del buf365  # reuse
    cpp_fused_index_add_51(c_void_p(buf411.data_ptr()))
    aten.index_put_(buf411, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf406, (1179648, ), (1, ), 0), True)
    buf414 = reinterpret_tensor(buf364, (786432, ), (1, ), 0); del buf364  # reuse
    buf417 = reinterpret_tensor(buf336, (1024, 1, 768), (768, 768, 1), 0); del buf336  # reuse
    buf418 = buf333; del buf333  # reuse
    buf420 = reinterpret_tensor(buf276, (1024, 768), (768, 1), 0); del buf276  # reuse
    cpp_fused_as_strided_scatter_div_view_52(c_void_p(buf411.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf420.data_ptr()))
    buf421 = reinterpret_tensor(buf417, (1024, 768), (768, 1), 0); del buf417  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf420, permute_1027, out=buf421)
    del permute_1027
    buf422 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf420, (768, 1024), (1, 768), 0), view_525, out=buf422)
    buf423 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_53(c_void_p(buf420.data_ptr()), c_void_p(buf423.data_ptr()))
    buf424 = buf420; del buf420  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf407, (1024, 768), (768, 1), 0), permute_1031, out=buf424)
    del permute_1031
    buf425 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf407, (768, 1024), (1, 768), 0), view_525, out=buf425)
    buf426 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_54(c_void_p(buf407.data_ptr()), c_void_p(buf426.data_ptr()))
    buf427 = reinterpret_tensor(buf407, (1024, 768), (768, 1), 0); del buf407  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf418, permute_1040, out=buf427)
    del permute_1040
    buf428 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf418, (768, 1024), (1, 768), 0), view_525, out=buf428)
    del view_525
    buf429 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf430 = reinterpret_tensor(buf414, (1, 1024, 768), (786432, 768, 1), 0); del buf414  # reuse
    buf431 = buf360; del buf360  # reuse
    buf432 = buf359; del buf359  # reuse
    buf433 = buf430; del buf430  # reuse
    buf434 = empty((768, ), device='cpu', dtype=torch.float32)
    buf435 = empty((768, ), device='cpu', dtype=torch.float32)
    buf436 = reinterpret_tensor(buf411, (1, 1024, 768), (786432, 768, 1), 0); del buf411  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_55(c_void_p(buf433.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(mul_54.data_ptr()), c_void_p(div_135.data_ptr()), c_void_p(getitem_67.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()))
    del div_135
    del getitem_67
    del mul_54
    del primals_111
    buf437 = reinterpret_tensor(buf355, (1024, 3072), (3072, 1), 0); del buf355  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf436, (1024, 768), (768, 1), 0), permute_1046, out=buf437)
    del permute_1046
    buf438 = reinterpret_tensor(buf369, (768, 3072), (3072, 1), 0); del buf369  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf436, (768, 1024), (1, 768), 0), view_523, out=buf438)
    del view_523
    buf439 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf440 = reinterpret_tensor(buf437, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf437  # reuse
    cpp_fused_gelu_gelu_backward_sum_56(c_void_p(buf440.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(addmm_40.data_ptr()), c_void_p(buf439.data_ptr()))
    del addmm_40
    buf441 = reinterpret_tensor(buf436, (1024, 768), (768, 1), 0); del buf436  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf440, (1024, 3072), (3072, 1), 0), permute_1050, out=buf441)
    del permute_1050
    buf442 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf440, (3072, 1024), (1, 3072), 0), view_521, out=buf442)
    del view_521
    buf443 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf444 = buf432; del buf432  # reuse
    buf445 = buf431; del buf431  # reuse
    buf446 = reinterpret_tensor(buf427, (1, 1024, 768), (786432, 768, 1), 0); del buf427  # reuse
    buf447 = empty((768, ), device='cpu', dtype=torch.float32)
    buf448 = empty((768, ), device='cpu', dtype=torch.float32)
    buf449 = reinterpret_tensor(buf424, (1, 1024, 768), (786432, 768, 1), 0); del buf424  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_57(c_void_p(buf440.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(mul_49.data_ptr()), c_void_p(div_136.data_ptr()), c_void_p(getitem_63.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf449.data_ptr()))
    del div_136
    del getitem_63
    del mul_49
    del primals_105
    buf450 = buf441; del buf441  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf449, (1024, 768), (768, 1), 0), permute_1054, out=buf450)
    del permute_1054
    buf451 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf449, (768, 1024), (1, 768), 0), view_519, out=buf451)
    del view_519
    buf452 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf453 = reinterpret_tensor(buf433, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf433  # reuse
    cpp_fused_clone_sum_58(c_void_p(buf449.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()))
    buf454 = empty((48, 768, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1062, reinterpret_tensor(buf453, (48, 256, 64), (16384, 64, 1), 0), out=buf454)
    del permute_1062
    buf455 = reinterpret_tensor(buf404, (48, 256, 768), (196608, 768, 1), 0); del buf404  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf453, (48, 256, 64), (16384, 64, 1), 0), permute_1063, out=buf455)
    del permute_1063
    buf456 = buf371; del buf371  # reuse
    cpp_fused_index_add_new_zeros_59(c_void_p(buf456.data_ptr()))
    aten.index_put_(buf456, [reinterpret_tensor(buf27, (2359296, ), (1, ), 0)], reinterpret_tensor(buf454, (2359296, ), (1, ), 0), True)
    buf459 = buf374; del buf374  # reuse
    buf460 = reinterpret_tensor(buf400, (1024, 12, 513), (513, 525312, 1), 0); del buf400  # reuse
    buf461 = reinterpret_tensor(buf397, (6303744, ), (1, ), 0); del buf397  # reuse
    buf464 = reinterpret_tensor(buf394, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf394  # reuse
    buf465 = buf392; del buf392  # reuse
    buf468 = buf390; del buf390  # reuse
    buf470 = reinterpret_tensor(buf386, (1024, 12, 513), (513, 525312, 1), 0); del buf386  # reuse
    buf471 = reinterpret_tensor(buf385, (6303744, ), (1, ), 0); del buf385  # reuse
    buf473 = buf383; del buf383  # reuse
    buf475 = buf380; del buf380  # reuse
    buf477 = buf473; del buf473  # reuse
    buf479 = reinterpret_tensor(buf379, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf379  # reuse
    buf482 = reinterpret_tensor(buf376, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf376  # reuse
    buf485 = reinterpret_tensor(buf375, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf375  # reuse
    buf487 = buf402; del buf402  # reuse
    buf489 = reinterpret_tensor(buf370, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf370  # reuse
    cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_60(c_void_p(buf477.data_ptr()), c_void_p(unsqueeze_16.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(getitem_61.data_ptr()), c_void_p(alias_17.data_ptr()), c_void_p(rev_1.data_ptr()), c_void_p(slice_64.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf489.data_ptr()))
    del alias_17
    del getitem_61
    buf490 = reinterpret_tensor(buf406, (36, 64, 512), (32768, 512, 1), 0); del buf406  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1073, reinterpret_tensor(buf489, (36, 512, 512), (262144, 512, 1), 0), out=buf490)
    del permute_1073
    buf491 = reinterpret_tensor(buf408, (36, 512, 64), (32768, 64, 1), 0); del buf408  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf489, (36, 512, 512), (262144, 512, 1), 0), permute_1074, out=buf491)
    del permute_1074
    buf492 = reinterpret_tensor(buf453, (786432, ), (1, ), 0); del buf453  # reuse
    buf493 = reinterpret_tensor(buf405, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf405  # reuse
    cpp_fused_clone_index_add_61(c_void_p(buf490.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf493.data_ptr()))
    aten.index_put_(buf492, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf493, (1179648, ), (1, ), 0), True)
    buf496 = reinterpret_tensor(buf450, (786432, ), (1, ), 0); del buf450  # reuse
    cpp_fused_index_add_62(c_void_p(buf496.data_ptr()))
    aten.index_put_(buf496, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf491, (1179648, ), (1, ), 0), True)
    buf499 = reinterpret_tensor(buf449, (786432, ), (1, ), 0); del buf449  # reuse
    buf502 = reinterpret_tensor(buf421, (1024, 1, 768), (768, 768, 1), 0); del buf421  # reuse
    buf503 = buf418; del buf418  # reuse
    buf505 = reinterpret_tensor(buf361, (1024, 768), (768, 1), 0); del buf361  # reuse
    cpp_fused_as_strided_scatter_div_view_63(c_void_p(buf496.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf505.data_ptr()))
    buf506 = reinterpret_tensor(buf502, (1024, 768), (768, 1), 0); del buf502  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf505, permute_1085, out=buf506)
    del permute_1085
    buf507 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf505, (768, 1024), (1, 768), 0), view_450, out=buf507)
    buf508 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_64(c_void_p(buf505.data_ptr()), c_void_p(buf508.data_ptr()))
    buf509 = buf505; del buf505  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf492, (1024, 768), (768, 1), 0), permute_1089, out=buf509)
    del permute_1089
    buf510 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf492, (768, 1024), (1, 768), 0), view_450, out=buf510)
    buf511 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_65(c_void_p(buf492.data_ptr()), c_void_p(buf511.data_ptr()))
    buf512 = reinterpret_tensor(buf492, (1024, 768), (768, 1), 0); del buf492  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf503, permute_1098, out=buf512)
    del permute_1098
    buf513 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf503, (768, 1024), (1, 768), 0), view_450, out=buf513)
    del view_450
    buf514 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf515 = reinterpret_tensor(buf499, (1, 1024, 768), (786432, 768, 1), 0); del buf499  # reuse
    buf516 = buf445; del buf445  # reuse
    buf517 = buf444; del buf444  # reuse
    buf518 = buf515; del buf515  # reuse
    buf519 = empty((768, ), device='cpu', dtype=torch.float32)
    buf520 = empty((768, ), device='cpu', dtype=torch.float32)
    buf521 = reinterpret_tensor(buf496, (1, 1024, 768), (786432, 768, 1), 0); del buf496  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_66(c_void_p(buf518.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(mul_46.data_ptr()), c_void_p(div_138.data_ptr()), c_void_p(getitem_57.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf516.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf521.data_ptr()))
    del div_138
    del getitem_57
    del mul_46
    del primals_95
    buf522 = reinterpret_tensor(buf440, (1024, 3072), (3072, 1), 0); del buf440  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf521, (1024, 768), (768, 1), 0), permute_1104, out=buf522)
    del permute_1104
    buf523 = reinterpret_tensor(buf454, (768, 3072), (3072, 1), 0); del buf454  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf521, (768, 1024), (1, 768), 0), view_448, out=buf523)
    del view_448
    buf524 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf525 = reinterpret_tensor(buf522, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf522  # reuse
    cpp_fused_gelu_gelu_backward_sum_67(c_void_p(buf525.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(buf524.data_ptr()))
    del addmm_34
    buf526 = reinterpret_tensor(buf521, (1024, 768), (768, 1), 0); del buf521  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf525, (1024, 3072), (3072, 1), 0), permute_1108, out=buf526)
    del permute_1108
    buf527 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf525, (3072, 1024), (1, 3072), 0), view_446, out=buf527)
    del view_446
    buf528 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf529 = buf517; del buf517  # reuse
    buf530 = buf516; del buf516  # reuse
    buf531 = reinterpret_tensor(buf512, (1, 1024, 768), (786432, 768, 1), 0); del buf512  # reuse
    buf532 = empty((768, ), device='cpu', dtype=torch.float32)
    buf533 = empty((768, ), device='cpu', dtype=torch.float32)
    buf534 = reinterpret_tensor(buf509, (1, 1024, 768), (786432, 768, 1), 0); del buf509  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_68(c_void_p(buf525.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(mul_41.data_ptr()), c_void_p(div_139.data_ptr()), c_void_p(getitem_53.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf534.data_ptr()))
    del div_139
    del getitem_53
    del mul_41
    del primals_89
    buf535 = buf526; del buf526  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf534, (1024, 768), (768, 1), 0), permute_1112, out=buf535)
    del permute_1112
    buf536 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf534, (768, 1024), (1, 768), 0), view_444, out=buf536)
    del view_444
    buf537 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf538 = reinterpret_tensor(buf518, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf518  # reuse
    cpp_fused_clone_sum_69(c_void_p(buf534.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf538.data_ptr()))
    buf539 = empty((48, 768, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1120, reinterpret_tensor(buf538, (48, 256, 64), (16384, 64, 1), 0), out=buf539)
    del permute_1120
    buf540 = reinterpret_tensor(buf489, (48, 256, 768), (196608, 768, 1), 0); del buf489  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf538, (48, 256, 64), (16384, 64, 1), 0), permute_1121, out=buf540)
    del permute_1121
    buf541 = buf456; del buf456  # reuse
    cpp_fused_index_add_new_zeros_70(c_void_p(buf541.data_ptr()))
    aten.index_put_(buf541, [reinterpret_tensor(buf27, (2359296, ), (1, ), 0)], reinterpret_tensor(buf539, (2359296, ), (1, ), 0), True)
    buf544 = buf459; del buf459  # reuse
    buf545 = reinterpret_tensor(buf485, (1024, 12, 513), (513, 525312, 1), 0); del buf485  # reuse
    buf546 = reinterpret_tensor(buf482, (6303744, ), (1, ), 0); del buf482  # reuse
    buf549 = reinterpret_tensor(buf479, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf479  # reuse
    buf550 = buf477; del buf477  # reuse
    buf553 = buf475; del buf475  # reuse
    buf555 = reinterpret_tensor(buf471, (1024, 12, 513), (513, 525312, 1), 0); del buf471  # reuse
    buf556 = reinterpret_tensor(buf470, (6303744, ), (1, ), 0); del buf470  # reuse
    buf558 = buf468; del buf468  # reuse
    buf560 = buf465; del buf465  # reuse
    buf562 = buf558; del buf558  # reuse
    buf564 = reinterpret_tensor(buf464, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf464  # reuse
    buf567 = reinterpret_tensor(buf461, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf461  # reuse
    buf570 = reinterpret_tensor(buf460, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf460  # reuse
    buf572 = buf487; del buf487  # reuse
    buf574 = reinterpret_tensor(buf455, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf455  # reuse
    cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_71(c_void_p(buf562.data_ptr()), c_void_p(unsqueeze_16.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(getitem_51.data_ptr()), c_void_p(alias_18.data_ptr()), c_void_p(rev_1.data_ptr()), c_void_p(slice_64.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf555.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(buf574.data_ptr()))
    del alias_18
    del getitem_51
    buf575 = reinterpret_tensor(buf491, (36, 64, 512), (32768, 512, 1), 0); del buf491  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1131, reinterpret_tensor(buf574, (36, 512, 512), (262144, 512, 1), 0), out=buf575)
    del permute_1131
    buf576 = reinterpret_tensor(buf493, (36, 512, 64), (32768, 64, 1), 0); del buf493  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf574, (36, 512, 512), (262144, 512, 1), 0), permute_1132, out=buf576)
    del permute_1132
    buf577 = reinterpret_tensor(buf538, (786432, ), (1, ), 0); del buf538  # reuse
    buf578 = reinterpret_tensor(buf490, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf490  # reuse
    cpp_fused_clone_index_add_72(c_void_p(buf575.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(buf578.data_ptr()))
    aten.index_put_(buf577, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf578, (1179648, ), (1, ), 0), True)
    buf581 = reinterpret_tensor(buf535, (786432, ), (1, ), 0); del buf535  # reuse
    cpp_fused_index_add_73(c_void_p(buf581.data_ptr()))
    aten.index_put_(buf581, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf576, (1179648, ), (1, ), 0), True)
    buf584 = reinterpret_tensor(buf534, (786432, ), (1, ), 0); del buf534  # reuse
    buf587 = reinterpret_tensor(buf506, (1024, 1, 768), (768, 768, 1), 0); del buf506  # reuse
    buf588 = buf503; del buf503  # reuse
    buf590 = reinterpret_tensor(buf446, (1024, 768), (768, 1), 0); del buf446  # reuse
    cpp_fused_as_strided_scatter_div_view_74(c_void_p(buf581.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(buf590.data_ptr()))
    buf591 = reinterpret_tensor(buf587, (1024, 768), (768, 1), 0); del buf587  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf590, permute_1143, out=buf591)
    del permute_1143
    buf592 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf590, (768, 1024), (1, 768), 0), view_375, out=buf592)
    buf593 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_75(c_void_p(buf590.data_ptr()), c_void_p(buf593.data_ptr()))
    buf594 = buf590; del buf590  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf577, (1024, 768), (768, 1), 0), permute_1147, out=buf594)
    del permute_1147
    buf595 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf577, (768, 1024), (1, 768), 0), view_375, out=buf595)
    buf596 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_76(c_void_p(buf577.data_ptr()), c_void_p(buf596.data_ptr()))
    buf597 = reinterpret_tensor(buf577, (1024, 768), (768, 1), 0); del buf577  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf588, permute_1156, out=buf597)
    del permute_1156
    buf598 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf588, (768, 1024), (1, 768), 0), view_375, out=buf598)
    del view_375
    buf599 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf600 = reinterpret_tensor(buf584, (1, 1024, 768), (786432, 768, 1), 0); del buf584  # reuse
    buf601 = buf530; del buf530  # reuse
    buf602 = buf529; del buf529  # reuse
    buf603 = buf600; del buf600  # reuse
    buf604 = empty((768, ), device='cpu', dtype=torch.float32)
    buf605 = empty((768, ), device='cpu', dtype=torch.float32)
    buf606 = reinterpret_tensor(buf581, (1, 1024, 768), (786432, 768, 1), 0); del buf581  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_77(c_void_p(buf603.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(mul_38.data_ptr()), c_void_p(div_141.data_ptr()), c_void_p(getitem_47.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf604.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf606.data_ptr()))
    del div_141
    del getitem_47
    del mul_38
    del primals_79
    buf607 = reinterpret_tensor(buf525, (1024, 3072), (3072, 1), 0); del buf525  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf606, (1024, 768), (768, 1), 0), permute_1162, out=buf607)
    del permute_1162
    buf608 = reinterpret_tensor(buf539, (768, 3072), (3072, 1), 0); del buf539  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf606, (768, 1024), (1, 768), 0), view_373, out=buf608)
    del view_373
    buf609 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf610 = reinterpret_tensor(buf607, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf607  # reuse
    cpp_fused_gelu_gelu_backward_sum_78(c_void_p(buf610.data_ptr()), c_void_p(buf606.data_ptr()), c_void_p(addmm_28.data_ptr()), c_void_p(buf609.data_ptr()))
    del addmm_28
    buf611 = reinterpret_tensor(buf606, (1024, 768), (768, 1), 0); del buf606  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf610, (1024, 3072), (3072, 1), 0), permute_1166, out=buf611)
    del permute_1166
    buf612 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf610, (3072, 1024), (1, 3072), 0), view_371, out=buf612)
    del view_371
    buf613 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf614 = buf602; del buf602  # reuse
    buf615 = buf601; del buf601  # reuse
    buf616 = reinterpret_tensor(buf597, (1, 1024, 768), (786432, 768, 1), 0); del buf597  # reuse
    buf617 = empty((768, ), device='cpu', dtype=torch.float32)
    buf618 = empty((768, ), device='cpu', dtype=torch.float32)
    buf619 = reinterpret_tensor(buf594, (1, 1024, 768), (786432, 768, 1), 0); del buf594  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_79(c_void_p(buf610.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(mul_33.data_ptr()), c_void_p(div_142.data_ptr()), c_void_p(getitem_43.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf615.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(buf618.data_ptr()), c_void_p(buf619.data_ptr()))
    del div_142
    del getitem_43
    del mul_33
    del primals_73
    buf620 = buf611; del buf611  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf619, (1024, 768), (768, 1), 0), permute_1170, out=buf620)
    del permute_1170
    buf621 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf619, (768, 1024), (1, 768), 0), view_369, out=buf621)
    del view_369
    buf622 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf623 = reinterpret_tensor(buf603, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf603  # reuse
    cpp_fused_clone_sum_80(c_void_p(buf619.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(buf622.data_ptr()), c_void_p(buf623.data_ptr()))
    buf624 = empty((48, 768, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1178, reinterpret_tensor(buf623, (48, 256, 64), (16384, 64, 1), 0), out=buf624)
    del permute_1178
    buf625 = reinterpret_tensor(buf574, (48, 256, 768), (196608, 768, 1), 0); del buf574  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf623, (48, 256, 64), (16384, 64, 1), 0), permute_1179, out=buf625)
    del permute_1179
    buf626 = buf541; del buf541  # reuse
    cpp_fused_index_add_new_zeros_81(c_void_p(buf626.data_ptr()))
    aten.index_put_(buf626, [reinterpret_tensor(buf27, (2359296, ), (1, ), 0)], reinterpret_tensor(buf624, (2359296, ), (1, ), 0), True)
    buf629 = buf544; del buf544  # reuse
    buf630 = reinterpret_tensor(buf570, (1024, 12, 513), (513, 525312, 1), 0); del buf570  # reuse
    buf631 = reinterpret_tensor(buf567, (6303744, ), (1, ), 0); del buf567  # reuse
    buf634 = reinterpret_tensor(buf564, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf564  # reuse
    buf635 = buf562; del buf562  # reuse
    buf638 = buf560; del buf560  # reuse
    buf640 = reinterpret_tensor(buf556, (1024, 12, 513), (513, 525312, 1), 0); del buf556  # reuse
    buf641 = reinterpret_tensor(buf555, (6303744, ), (1, ), 0); del buf555  # reuse
    buf643 = buf553; del buf553  # reuse
    buf645 = buf550; del buf550  # reuse
    buf647 = buf643; del buf643  # reuse
    buf649 = reinterpret_tensor(buf549, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf549  # reuse
    buf652 = reinterpret_tensor(buf546, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf546  # reuse
    buf655 = reinterpret_tensor(buf545, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf545  # reuse
    buf657 = buf572; del buf572  # reuse
    buf659 = reinterpret_tensor(buf540, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf540  # reuse
    cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_82(c_void_p(buf647.data_ptr()), c_void_p(unsqueeze_16.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(getitem_41.data_ptr()), c_void_p(alias_19.data_ptr()), c_void_p(rev_1.data_ptr()), c_void_p(slice_64.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(buf631.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(buf635.data_ptr()), c_void_p(buf638.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(buf641.data_ptr()), c_void_p(buf645.data_ptr()), c_void_p(buf649.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(buf655.data_ptr()), c_void_p(buf657.data_ptr()), c_void_p(buf659.data_ptr()))
    del alias_19
    del getitem_41
    buf660 = reinterpret_tensor(buf576, (36, 64, 512), (32768, 512, 1), 0); del buf576  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1189, reinterpret_tensor(buf659, (36, 512, 512), (262144, 512, 1), 0), out=buf660)
    del permute_1189
    buf661 = reinterpret_tensor(buf578, (36, 512, 64), (32768, 64, 1), 0); del buf578  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf659, (36, 512, 512), (262144, 512, 1), 0), permute_1190, out=buf661)
    del permute_1190
    buf662 = reinterpret_tensor(buf623, (786432, ), (1, ), 0); del buf623  # reuse
    buf663 = reinterpret_tensor(buf575, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf575  # reuse
    cpp_fused_clone_index_add_83(c_void_p(buf660.data_ptr()), c_void_p(buf662.data_ptr()), c_void_p(buf663.data_ptr()))
    aten.index_put_(buf662, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf663, (1179648, ), (1, ), 0), True)
    buf666 = reinterpret_tensor(buf620, (786432, ), (1, ), 0); del buf620  # reuse
    cpp_fused_index_add_84(c_void_p(buf666.data_ptr()))
    aten.index_put_(buf666, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf661, (1179648, ), (1, ), 0), True)
    buf669 = reinterpret_tensor(buf619, (786432, ), (1, ), 0); del buf619  # reuse
    buf672 = reinterpret_tensor(buf591, (1024, 1, 768), (768, 768, 1), 0); del buf591  # reuse
    buf673 = buf588; del buf588  # reuse
    buf675 = reinterpret_tensor(buf531, (1024, 768), (768, 1), 0); del buf531  # reuse
    cpp_fused_as_strided_scatter_div_view_85(c_void_p(buf666.data_ptr()), c_void_p(buf626.data_ptr()), c_void_p(buf669.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(buf675.data_ptr()))
    buf676 = reinterpret_tensor(buf672, (1024, 768), (768, 1), 0); del buf672  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf675, permute_1201, out=buf676)
    del permute_1201
    buf677 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf675, (768, 1024), (1, 768), 0), view_300, out=buf677)
    buf678 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_86(c_void_p(buf675.data_ptr()), c_void_p(buf678.data_ptr()))
    buf679 = buf675; del buf675  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf662, (1024, 768), (768, 1), 0), permute_1205, out=buf679)
    del permute_1205
    buf680 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf662, (768, 1024), (1, 768), 0), view_300, out=buf680)
    buf681 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_87(c_void_p(buf662.data_ptr()), c_void_p(buf681.data_ptr()))
    buf682 = reinterpret_tensor(buf662, (1024, 768), (768, 1), 0); del buf662  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf673, permute_1214, out=buf682)
    del permute_1214
    buf683 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf673, (768, 1024), (1, 768), 0), view_300, out=buf683)
    del view_300
    buf684 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf685 = reinterpret_tensor(buf669, (1, 1024, 768), (786432, 768, 1), 0); del buf669  # reuse
    buf686 = buf615; del buf615  # reuse
    buf687 = buf614; del buf614  # reuse
    buf688 = buf685; del buf685  # reuse
    buf689 = empty((768, ), device='cpu', dtype=torch.float32)
    buf690 = empty((768, ), device='cpu', dtype=torch.float32)
    buf691 = reinterpret_tensor(buf666, (1, 1024, 768), (786432, 768, 1), 0); del buf666  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_88(c_void_p(buf688.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(buf676.data_ptr()), c_void_p(buf679.data_ptr()), c_void_p(buf682.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(mul_30.data_ptr()), c_void_p(div_144.data_ptr()), c_void_p(getitem_37.data_ptr()), c_void_p(buf684.data_ptr()), c_void_p(buf686.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(buf689.data_ptr()), c_void_p(buf690.data_ptr()), c_void_p(buf691.data_ptr()))
    del div_144
    del getitem_37
    del mul_30
    del primals_63
    buf692 = reinterpret_tensor(buf610, (1024, 3072), (3072, 1), 0); del buf610  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf691, (1024, 768), (768, 1), 0), permute_1220, out=buf692)
    del permute_1220
    buf693 = reinterpret_tensor(buf624, (768, 3072), (3072, 1), 0); del buf624  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf691, (768, 1024), (1, 768), 0), view_298, out=buf693)
    del view_298
    buf694 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf695 = reinterpret_tensor(buf692, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf692  # reuse
    cpp_fused_gelu_gelu_backward_sum_89(c_void_p(buf695.data_ptr()), c_void_p(buf691.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(buf694.data_ptr()))
    del addmm_22
    buf696 = reinterpret_tensor(buf691, (1024, 768), (768, 1), 0); del buf691  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf695, (1024, 3072), (3072, 1), 0), permute_1224, out=buf696)
    del permute_1224
    buf697 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf695, (3072, 1024), (1, 3072), 0), view_296, out=buf697)
    del view_296
    buf698 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf699 = buf687; del buf687  # reuse
    buf700 = buf686; del buf686  # reuse
    buf701 = reinterpret_tensor(buf682, (1, 1024, 768), (786432, 768, 1), 0); del buf682  # reuse
    buf702 = empty((768, ), device='cpu', dtype=torch.float32)
    buf703 = empty((768, ), device='cpu', dtype=torch.float32)
    buf704 = reinterpret_tensor(buf679, (1, 1024, 768), (786432, 768, 1), 0); del buf679  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_90(c_void_p(buf695.data_ptr()), c_void_p(buf688.data_ptr()), c_void_p(buf696.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(mul_25.data_ptr()), c_void_p(div_145.data_ptr()), c_void_p(getitem_33.data_ptr()), c_void_p(buf698.data_ptr()), c_void_p(buf699.data_ptr()), c_void_p(buf700.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf702.data_ptr()), c_void_p(buf703.data_ptr()), c_void_p(buf704.data_ptr()))
    del div_145
    del getitem_33
    del mul_25
    del primals_57
    buf705 = buf696; del buf696  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf704, (1024, 768), (768, 1), 0), permute_1228, out=buf705)
    del permute_1228
    buf706 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf704, (768, 1024), (1, 768), 0), view_294, out=buf706)
    del view_294
    buf707 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf708 = reinterpret_tensor(buf688, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf688  # reuse
    cpp_fused_clone_sum_91(c_void_p(buf704.data_ptr()), c_void_p(buf705.data_ptr()), c_void_p(buf707.data_ptr()), c_void_p(buf708.data_ptr()))
    buf709 = empty((48, 768, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1236, reinterpret_tensor(buf708, (48, 256, 64), (16384, 64, 1), 0), out=buf709)
    del permute_1236
    buf710 = reinterpret_tensor(buf659, (48, 256, 768), (196608, 768, 1), 0); del buf659  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf708, (48, 256, 64), (16384, 64, 1), 0), permute_1237, out=buf710)
    del permute_1237
    buf711 = buf626; del buf626  # reuse
    cpp_fused_index_add_new_zeros_92(c_void_p(buf711.data_ptr()))
    aten.index_put_(buf711, [reinterpret_tensor(buf27, (2359296, ), (1, ), 0)], reinterpret_tensor(buf709, (2359296, ), (1, ), 0), True)
    buf714 = buf629; del buf629  # reuse
    buf715 = reinterpret_tensor(buf655, (1024, 12, 513), (513, 525312, 1), 0); del buf655  # reuse
    buf716 = reinterpret_tensor(buf652, (6303744, ), (1, ), 0); del buf652  # reuse
    buf719 = reinterpret_tensor(buf649, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf649  # reuse
    buf720 = buf647; del buf647  # reuse
    buf723 = buf645; del buf645  # reuse
    buf725 = reinterpret_tensor(buf641, (1024, 12, 513), (513, 525312, 1), 0); del buf641  # reuse
    buf726 = reinterpret_tensor(buf640, (6303744, ), (1, ), 0); del buf640  # reuse
    buf728 = buf638; del buf638  # reuse
    buf730 = buf635; del buf635  # reuse
    buf732 = buf728; del buf728  # reuse
    buf734 = reinterpret_tensor(buf634, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf634  # reuse
    buf737 = reinterpret_tensor(buf631, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf631  # reuse
    buf740 = reinterpret_tensor(buf630, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf630  # reuse
    buf742 = buf657; del buf657  # reuse
    buf744 = reinterpret_tensor(buf625, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf625  # reuse
    cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_93(c_void_p(buf732.data_ptr()), c_void_p(unsqueeze_16.data_ptr()), c_void_p(buf710.data_ptr()), c_void_p(getitem_31.data_ptr()), c_void_p(alias_20.data_ptr()), c_void_p(rev_1.data_ptr()), c_void_p(slice_64.data_ptr()), c_void_p(buf714.data_ptr()), c_void_p(buf715.data_ptr()), c_void_p(buf716.data_ptr()), c_void_p(buf719.data_ptr()), c_void_p(buf720.data_ptr()), c_void_p(buf723.data_ptr()), c_void_p(buf725.data_ptr()), c_void_p(buf726.data_ptr()), c_void_p(buf730.data_ptr()), c_void_p(buf734.data_ptr()), c_void_p(buf737.data_ptr()), c_void_p(buf740.data_ptr()), c_void_p(buf742.data_ptr()), c_void_p(buf744.data_ptr()))
    del alias_20
    del getitem_31
    buf745 = reinterpret_tensor(buf661, (36, 64, 512), (32768, 512, 1), 0); del buf661  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1247, reinterpret_tensor(buf744, (36, 512, 512), (262144, 512, 1), 0), out=buf745)
    del permute_1247
    buf746 = reinterpret_tensor(buf663, (36, 512, 64), (32768, 64, 1), 0); del buf663  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf744, (36, 512, 512), (262144, 512, 1), 0), permute_1248, out=buf746)
    del permute_1248
    buf747 = reinterpret_tensor(buf708, (786432, ), (1, ), 0); del buf708  # reuse
    buf748 = reinterpret_tensor(buf660, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf660  # reuse
    cpp_fused_clone_index_add_94(c_void_p(buf745.data_ptr()), c_void_p(buf747.data_ptr()), c_void_p(buf748.data_ptr()))
    aten.index_put_(buf747, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf748, (1179648, ), (1, ), 0), True)
    buf751 = reinterpret_tensor(buf705, (786432, ), (1, ), 0); del buf705  # reuse
    cpp_fused_index_add_95(c_void_p(buf751.data_ptr()))
    aten.index_put_(buf751, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf746, (1179648, ), (1, ), 0), True)
    buf754 = reinterpret_tensor(buf704, (786432, ), (1, ), 0); del buf704  # reuse
    buf757 = reinterpret_tensor(buf676, (1024, 1, 768), (768, 768, 1), 0); del buf676  # reuse
    buf758 = buf673; del buf673  # reuse
    buf760 = reinterpret_tensor(buf616, (1024, 768), (768, 1), 0); del buf616  # reuse
    cpp_fused_as_strided_scatter_div_view_96(c_void_p(buf751.data_ptr()), c_void_p(buf711.data_ptr()), c_void_p(buf754.data_ptr()), c_void_p(buf757.data_ptr()), c_void_p(buf758.data_ptr()), c_void_p(buf760.data_ptr()))
    buf761 = reinterpret_tensor(buf757, (1024, 768), (768, 1), 0); del buf757  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf760, permute_1259, out=buf761)
    del permute_1259
    buf762 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf760, (768, 1024), (1, 768), 0), view_225, out=buf762)
    buf763 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_97(c_void_p(buf760.data_ptr()), c_void_p(buf763.data_ptr()))
    buf764 = buf760; del buf760  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf747, (1024, 768), (768, 1), 0), permute_1263, out=buf764)
    del permute_1263
    buf765 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf747, (768, 1024), (1, 768), 0), view_225, out=buf765)
    buf766 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_98(c_void_p(buf747.data_ptr()), c_void_p(buf766.data_ptr()))
    buf767 = reinterpret_tensor(buf747, (1024, 768), (768, 1), 0); del buf747  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf758, permute_1272, out=buf767)
    del permute_1272
    buf768 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf758, (768, 1024), (1, 768), 0), view_225, out=buf768)
    del view_225
    buf769 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf770 = reinterpret_tensor(buf754, (1, 1024, 768), (786432, 768, 1), 0); del buf754  # reuse
    buf771 = buf700; del buf700  # reuse
    buf772 = buf699; del buf699  # reuse
    buf773 = buf770; del buf770  # reuse
    buf774 = empty((768, ), device='cpu', dtype=torch.float32)
    buf775 = empty((768, ), device='cpu', dtype=torch.float32)
    buf776 = reinterpret_tensor(buf751, (1, 1024, 768), (786432, 768, 1), 0); del buf751  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_99(c_void_p(buf773.data_ptr()), c_void_p(buf758.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf761.data_ptr()), c_void_p(buf764.data_ptr()), c_void_p(buf767.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(mul_22.data_ptr()), c_void_p(div_147.data_ptr()), c_void_p(getitem_27.data_ptr()), c_void_p(buf769.data_ptr()), c_void_p(buf771.data_ptr()), c_void_p(buf772.data_ptr()), c_void_p(buf774.data_ptr()), c_void_p(buf775.data_ptr()), c_void_p(buf776.data_ptr()))
    del div_147
    del getitem_27
    del mul_22
    del primals_47
    buf777 = reinterpret_tensor(buf695, (1024, 3072), (3072, 1), 0); del buf695  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf776, (1024, 768), (768, 1), 0), permute_1278, out=buf777)
    del permute_1278
    buf778 = reinterpret_tensor(buf709, (768, 3072), (3072, 1), 0); del buf709  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf776, (768, 1024), (1, 768), 0), view_223, out=buf778)
    del view_223
    buf779 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf780 = reinterpret_tensor(buf777, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf777  # reuse
    cpp_fused_gelu_gelu_backward_sum_100(c_void_p(buf780.data_ptr()), c_void_p(buf776.data_ptr()), c_void_p(addmm_16.data_ptr()), c_void_p(buf779.data_ptr()))
    del addmm_16
    buf781 = reinterpret_tensor(buf776, (1024, 768), (768, 1), 0); del buf776  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf780, (1024, 3072), (3072, 1), 0), permute_1282, out=buf781)
    del permute_1282
    buf782 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf780, (3072, 1024), (1, 3072), 0), view_221, out=buf782)
    del view_221
    buf783 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf784 = buf772; del buf772  # reuse
    buf785 = buf771; del buf771  # reuse
    buf786 = reinterpret_tensor(buf767, (1, 1024, 768), (786432, 768, 1), 0); del buf767  # reuse
    buf787 = empty((768, ), device='cpu', dtype=torch.float32)
    buf788 = empty((768, ), device='cpu', dtype=torch.float32)
    buf789 = reinterpret_tensor(buf764, (1, 1024, 768), (786432, 768, 1), 0); del buf764  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_101(c_void_p(buf780.data_ptr()), c_void_p(buf773.data_ptr()), c_void_p(buf781.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(mul_17.data_ptr()), c_void_p(div_148.data_ptr()), c_void_p(getitem_23.data_ptr()), c_void_p(buf783.data_ptr()), c_void_p(buf784.data_ptr()), c_void_p(buf785.data_ptr()), c_void_p(buf786.data_ptr()), c_void_p(buf787.data_ptr()), c_void_p(buf788.data_ptr()), c_void_p(buf789.data_ptr()))
    del div_148
    del getitem_23
    del mul_17
    del primals_41
    buf790 = buf781; del buf781  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf789, (1024, 768), (768, 1), 0), permute_1286, out=buf790)
    del permute_1286
    buf791 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf789, (768, 1024), (1, 768), 0), view_219, out=buf791)
    del view_219
    buf792 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf793 = reinterpret_tensor(buf773, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf773  # reuse
    cpp_fused_clone_sum_102(c_void_p(buf789.data_ptr()), c_void_p(buf790.data_ptr()), c_void_p(buf792.data_ptr()), c_void_p(buf793.data_ptr()))
    buf794 = empty((48, 768, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1294, reinterpret_tensor(buf793, (48, 256, 64), (16384, 64, 1), 0), out=buf794)
    del permute_1294
    buf795 = reinterpret_tensor(buf744, (48, 256, 768), (196608, 768, 1), 0); del buf744  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf793, (48, 256, 64), (16384, 64, 1), 0), permute_1295, out=buf795)
    del permute_1295
    buf796 = buf711; del buf711  # reuse
    cpp_fused_index_add_new_zeros_103(c_void_p(buf796.data_ptr()))
    aten.index_put_(buf796, [reinterpret_tensor(buf27, (2359296, ), (1, ), 0)], reinterpret_tensor(buf794, (2359296, ), (1, ), 0), True)
    buf799 = buf714; del buf714  # reuse
    buf800 = reinterpret_tensor(buf740, (1024, 12, 513), (513, 525312, 1), 0); del buf740  # reuse
    buf801 = reinterpret_tensor(buf737, (6303744, ), (1, ), 0); del buf737  # reuse
    buf804 = reinterpret_tensor(buf734, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf734  # reuse
    buf805 = buf732; del buf732  # reuse
    buf808 = buf730; del buf730  # reuse
    buf810 = reinterpret_tensor(buf726, (1024, 12, 513), (513, 525312, 1), 0); del buf726  # reuse
    buf811 = reinterpret_tensor(buf725, (6303744, ), (1, ), 0); del buf725  # reuse
    buf813 = buf723; del buf723  # reuse
    buf815 = buf720; del buf720  # reuse
    buf817 = buf813; del buf813  # reuse
    buf819 = reinterpret_tensor(buf719, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf719  # reuse
    buf822 = reinterpret_tensor(buf716, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf716  # reuse
    buf825 = reinterpret_tensor(buf715, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf715  # reuse
    buf827 = buf742; del buf742  # reuse
    buf829 = reinterpret_tensor(buf710, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf710  # reuse
    cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_104(c_void_p(buf817.data_ptr()), c_void_p(unsqueeze_16.data_ptr()), c_void_p(buf795.data_ptr()), c_void_p(getitem_21.data_ptr()), c_void_p(alias_21.data_ptr()), c_void_p(rev_1.data_ptr()), c_void_p(slice_64.data_ptr()), c_void_p(buf799.data_ptr()), c_void_p(buf800.data_ptr()), c_void_p(buf801.data_ptr()), c_void_p(buf804.data_ptr()), c_void_p(buf805.data_ptr()), c_void_p(buf808.data_ptr()), c_void_p(buf810.data_ptr()), c_void_p(buf811.data_ptr()), c_void_p(buf815.data_ptr()), c_void_p(buf819.data_ptr()), c_void_p(buf822.data_ptr()), c_void_p(buf825.data_ptr()), c_void_p(buf827.data_ptr()), c_void_p(buf829.data_ptr()))
    del alias_21
    del getitem_21
    buf830 = reinterpret_tensor(buf746, (36, 64, 512), (32768, 512, 1), 0); del buf746  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1305, reinterpret_tensor(buf829, (36, 512, 512), (262144, 512, 1), 0), out=buf830)
    del permute_1305
    buf831 = reinterpret_tensor(buf748, (36, 512, 64), (32768, 64, 1), 0); del buf748  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf829, (36, 512, 512), (262144, 512, 1), 0), permute_1306, out=buf831)
    del permute_1306
    buf832 = reinterpret_tensor(buf793, (786432, ), (1, ), 0); del buf793  # reuse
    buf833 = reinterpret_tensor(buf745, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf745  # reuse
    cpp_fused_clone_index_add_105(c_void_p(buf830.data_ptr()), c_void_p(buf832.data_ptr()), c_void_p(buf833.data_ptr()))
    aten.index_put_(buf832, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf833, (1179648, ), (1, ), 0), True)
    buf836 = reinterpret_tensor(buf790, (786432, ), (1, ), 0); del buf790  # reuse
    cpp_fused_index_add_106(c_void_p(buf836.data_ptr()))
    aten.index_put_(buf836, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf831, (1179648, ), (1, ), 0), True)
    buf839 = reinterpret_tensor(buf789, (786432, ), (1, ), 0); del buf789  # reuse
    buf842 = reinterpret_tensor(buf761, (1024, 1, 768), (768, 768, 1), 0); del buf761  # reuse
    buf843 = buf758; del buf758  # reuse
    buf845 = reinterpret_tensor(buf701, (1024, 768), (768, 1), 0); del buf701  # reuse
    cpp_fused_as_strided_scatter_div_view_107(c_void_p(buf836.data_ptr()), c_void_p(buf796.data_ptr()), c_void_p(buf839.data_ptr()), c_void_p(buf842.data_ptr()), c_void_p(buf843.data_ptr()), c_void_p(buf845.data_ptr()))
    buf846 = reinterpret_tensor(buf842, (1024, 768), (768, 1), 0); del buf842  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf845, permute_1317, out=buf846)
    del permute_1317
    buf847 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf845, (768, 1024), (1, 768), 0), view_150, out=buf847)
    buf848 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_108(c_void_p(buf845.data_ptr()), c_void_p(buf848.data_ptr()))
    buf849 = buf845; del buf845  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf832, (1024, 768), (768, 1), 0), permute_1321, out=buf849)
    del permute_1321
    buf850 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf832, (768, 1024), (1, 768), 0), view_150, out=buf850)
    buf851 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_109(c_void_p(buf832.data_ptr()), c_void_p(buf851.data_ptr()))
    buf852 = reinterpret_tensor(buf832, (1024, 768), (768, 1), 0); del buf832  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf843, permute_1330, out=buf852)
    del permute_1330
    buf853 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf843, (768, 1024), (1, 768), 0), view_150, out=buf853)
    del view_150
    buf854 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf855 = reinterpret_tensor(buf839, (1, 1024, 768), (786432, 768, 1), 0); del buf839  # reuse
    buf856 = buf785; del buf785  # reuse
    buf857 = buf784; del buf784  # reuse
    buf858 = buf855; del buf855  # reuse
    buf859 = empty((768, ), device='cpu', dtype=torch.float32)
    buf860 = empty((768, ), device='cpu', dtype=torch.float32)
    buf861 = reinterpret_tensor(buf836, (1, 1024, 768), (786432, 768, 1), 0); del buf836  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_110(c_void_p(buf858.data_ptr()), c_void_p(buf843.data_ptr()), c_void_p(buf786.data_ptr()), c_void_p(buf846.data_ptr()), c_void_p(buf849.data_ptr()), c_void_p(buf852.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(mul_14.data_ptr()), c_void_p(div_150.data_ptr()), c_void_p(getitem_17.data_ptr()), c_void_p(buf854.data_ptr()), c_void_p(buf856.data_ptr()), c_void_p(buf857.data_ptr()), c_void_p(buf859.data_ptr()), c_void_p(buf860.data_ptr()), c_void_p(buf861.data_ptr()))
    del div_150
    del getitem_17
    del mul_14
    del primals_31
    buf862 = reinterpret_tensor(buf780, (1024, 3072), (3072, 1), 0); del buf780  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf861, (1024, 768), (768, 1), 0), permute_1336, out=buf862)
    del permute_1336
    buf863 = reinterpret_tensor(buf794, (768, 3072), (3072, 1), 0); del buf794  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf861, (768, 1024), (1, 768), 0), view_148, out=buf863)
    del view_148
    buf864 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf865 = reinterpret_tensor(buf862, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf862  # reuse
    cpp_fused_gelu_gelu_backward_sum_111(c_void_p(buf865.data_ptr()), c_void_p(buf861.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(buf864.data_ptr()))
    del addmm_10
    buf866 = reinterpret_tensor(buf861, (1024, 768), (768, 1), 0); del buf861  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf865, (1024, 3072), (3072, 1), 0), permute_1340, out=buf866)
    del permute_1340
    buf867 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf865, (3072, 1024), (1, 3072), 0), view_146, out=buf867)
    del view_146
    buf868 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf869 = buf857; del buf857  # reuse
    buf870 = buf856; del buf856  # reuse
    buf871 = reinterpret_tensor(buf852, (1, 1024, 768), (786432, 768, 1), 0); del buf852  # reuse
    buf872 = empty((768, ), device='cpu', dtype=torch.float32)
    buf873 = empty((768, ), device='cpu', dtype=torch.float32)
    buf874 = reinterpret_tensor(buf849, (1, 1024, 768), (786432, 768, 1), 0); del buf849  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_112(c_void_p(buf865.data_ptr()), c_void_p(buf858.data_ptr()), c_void_p(buf866.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(mul_9.data_ptr()), c_void_p(div_151.data_ptr()), c_void_p(getitem_13.data_ptr()), c_void_p(buf868.data_ptr()), c_void_p(buf869.data_ptr()), c_void_p(buf870.data_ptr()), c_void_p(buf871.data_ptr()), c_void_p(buf872.data_ptr()), c_void_p(buf873.data_ptr()), c_void_p(buf874.data_ptr()))
    del div_151
    del getitem_13
    del mul_9
    del primals_25
    buf875 = buf866; del buf866  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf874, (1024, 768), (768, 1), 0), permute_1344, out=buf875)
    del permute_1344
    buf876 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf874, (768, 1024), (1, 768), 0), view_144, out=buf876)
    del view_144
    buf877 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf878 = reinterpret_tensor(buf858, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf858  # reuse
    cpp_fused_clone_sum_113(c_void_p(buf874.data_ptr()), c_void_p(buf875.data_ptr()), c_void_p(buf877.data_ptr()), c_void_p(buf878.data_ptr()))
    buf879 = empty((48, 768, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1352, reinterpret_tensor(buf878, (48, 256, 64), (16384, 64, 1), 0), out=buf879)
    del permute_1352
    buf880 = reinterpret_tensor(buf829, (48, 256, 768), (196608, 768, 1), 0); del buf829  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf878, (48, 256, 64), (16384, 64, 1), 0), permute_1353, out=buf880)
    del permute_1353
    buf881 = buf796; del buf796  # reuse
    cpp_fused_index_add_new_zeros_114(c_void_p(buf881.data_ptr()))
    aten.index_put_(buf881, [reinterpret_tensor(buf27, (2359296, ), (1, ), 0)], reinterpret_tensor(buf879, (2359296, ), (1, ), 0), True)
    buf884 = buf799; del buf799  # reuse
    buf885 = reinterpret_tensor(buf825, (1024, 12, 513), (513, 525312, 1), 0); del buf825  # reuse
    buf886 = reinterpret_tensor(buf822, (6303744, ), (1, ), 0); del buf822  # reuse
    buf889 = reinterpret_tensor(buf819, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf819  # reuse
    buf890 = buf817; del buf817  # reuse
    buf893 = buf815; del buf815  # reuse
    buf895 = reinterpret_tensor(buf811, (1024, 12, 513), (513, 525312, 1), 0); del buf811  # reuse
    buf896 = reinterpret_tensor(buf810, (6303744, ), (1, ), 0); del buf810  # reuse
    buf898 = buf808; del buf808  # reuse
    buf900 = buf805; del buf805  # reuse
    buf902 = buf898; del buf898  # reuse
    buf904 = reinterpret_tensor(buf804, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf804  # reuse
    buf907 = reinterpret_tensor(buf801, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf801  # reuse
    buf910 = reinterpret_tensor(buf800, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf800  # reuse
    buf912 = buf827; del buf827  # reuse
    buf914 = reinterpret_tensor(buf795, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf795  # reuse
    cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_115(c_void_p(buf902.data_ptr()), c_void_p(unsqueeze_16.data_ptr()), c_void_p(buf880.data_ptr()), c_void_p(getitem_11.data_ptr()), c_void_p(alias_22.data_ptr()), c_void_p(rev_1.data_ptr()), c_void_p(slice_64.data_ptr()), c_void_p(buf884.data_ptr()), c_void_p(buf885.data_ptr()), c_void_p(buf886.data_ptr()), c_void_p(buf889.data_ptr()), c_void_p(buf890.data_ptr()), c_void_p(buf893.data_ptr()), c_void_p(buf895.data_ptr()), c_void_p(buf896.data_ptr()), c_void_p(buf900.data_ptr()), c_void_p(buf904.data_ptr()), c_void_p(buf907.data_ptr()), c_void_p(buf910.data_ptr()), c_void_p(buf912.data_ptr()), c_void_p(buf914.data_ptr()))
    del alias_22
    del getitem_11
    buf915 = reinterpret_tensor(buf831, (36, 64, 512), (32768, 512, 1), 0); del buf831  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1363, reinterpret_tensor(buf914, (36, 512, 512), (262144, 512, 1), 0), out=buf915)
    del permute_1363
    buf916 = reinterpret_tensor(buf833, (36, 512, 64), (32768, 64, 1), 0); del buf833  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf914, (36, 512, 512), (262144, 512, 1), 0), permute_1364, out=buf916)
    del permute_1364
    buf917 = reinterpret_tensor(buf878, (786432, ), (1, ), 0); del buf878  # reuse
    buf918 = reinterpret_tensor(buf830, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf830  # reuse
    cpp_fused_clone_index_add_116(c_void_p(buf915.data_ptr()), c_void_p(buf917.data_ptr()), c_void_p(buf918.data_ptr()))
    aten.index_put_(buf917, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf918, (1179648, ), (1, ), 0), True)
    buf921 = reinterpret_tensor(buf875, (786432, ), (1, ), 0); del buf875  # reuse
    cpp_fused_index_add_117(c_void_p(buf921.data_ptr()))
    aten.index_put_(buf921, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf916, (1179648, ), (1, ), 0), True)
    buf924 = reinterpret_tensor(buf874, (786432, ), (1, ), 0); del buf874  # reuse
    buf927 = reinterpret_tensor(buf846, (1024, 1, 768), (768, 768, 1), 0); del buf846  # reuse
    buf928 = buf843; del buf843  # reuse
    buf930 = reinterpret_tensor(buf786, (1024, 768), (768, 1), 0); del buf786  # reuse
    cpp_fused_as_strided_scatter_div_view_118(c_void_p(buf921.data_ptr()), c_void_p(buf881.data_ptr()), c_void_p(buf924.data_ptr()), c_void_p(buf927.data_ptr()), c_void_p(buf928.data_ptr()), c_void_p(buf930.data_ptr()))
    buf931 = reinterpret_tensor(buf927, (1024, 768), (768, 1), 0); del buf927  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf930, permute_1375, out=buf931)
    del permute_1375
    buf932 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf930, (768, 1024), (1, 768), 0), view_75, out=buf932)
    buf933 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_119(c_void_p(buf930.data_ptr()), c_void_p(buf933.data_ptr()))
    buf934 = buf930; del buf930  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf917, (1024, 768), (768, 1), 0), permute_1379, out=buf934)
    del permute_1379
    buf935 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf917, (768, 1024), (1, 768), 0), view_75, out=buf935)
    buf936 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_120(c_void_p(buf917.data_ptr()), c_void_p(buf936.data_ptr()))
    buf937 = reinterpret_tensor(buf917, (1024, 768), (768, 1), 0); del buf917  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf928, permute_1388, out=buf937)
    del permute_1388
    buf938 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf928, (768, 1024), (1, 768), 0), view_75, out=buf938)
    del view_75
    buf939 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf940 = reinterpret_tensor(buf924, (1, 1024, 768), (786432, 768, 1), 0); del buf924  # reuse
    buf941 = buf870; del buf870  # reuse
    buf942 = buf869; del buf869  # reuse
    buf943 = buf940; del buf940  # reuse
    buf944 = empty((768, ), device='cpu', dtype=torch.float32)
    buf945 = empty((768, ), device='cpu', dtype=torch.float32)
    buf946 = reinterpret_tensor(buf921, (1, 1024, 768), (786432, 768, 1), 0); del buf921  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_121(c_void_p(buf943.data_ptr()), c_void_p(buf928.data_ptr()), c_void_p(buf871.data_ptr()), c_void_p(buf931.data_ptr()), c_void_p(buf934.data_ptr()), c_void_p(buf937.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(mul_6.data_ptr()), c_void_p(div_153.data_ptr()), c_void_p(getitem_7.data_ptr()), c_void_p(buf939.data_ptr()), c_void_p(buf941.data_ptr()), c_void_p(buf942.data_ptr()), c_void_p(buf944.data_ptr()), c_void_p(buf945.data_ptr()), c_void_p(buf946.data_ptr()))
    del div_153
    del getitem_7
    del mul_6
    del primals_15
    buf947 = reinterpret_tensor(buf865, (1024, 3072), (3072, 1), 0); del buf865  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf946, (1024, 768), (768, 1), 0), permute_1394, out=buf947)
    del permute_1394
    buf948 = reinterpret_tensor(buf879, (768, 3072), (3072, 1), 0); del buf879  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf946, (768, 1024), (1, 768), 0), view_73, out=buf948)
    del view_73
    buf949 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf950 = reinterpret_tensor(buf947, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf947  # reuse
    cpp_fused_gelu_gelu_backward_sum_122(c_void_p(buf950.data_ptr()), c_void_p(buf946.data_ptr()), c_void_p(addmm_4.data_ptr()), c_void_p(buf949.data_ptr()))
    del addmm_4
    buf951 = reinterpret_tensor(buf946, (1024, 768), (768, 1), 0); del buf946  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf950, (1024, 3072), (3072, 1), 0), permute_1398, out=buf951)
    del permute_1398
    buf952 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf950, (3072, 1024), (1, 3072), 0), view_71, out=buf952)
    del view_71
    buf953 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf954 = buf942; del buf942  # reuse
    buf955 = buf941; del buf941  # reuse
    buf956 = reinterpret_tensor(buf937, (1, 1024, 768), (786432, 768, 1), 0); del buf937  # reuse
    buf957 = empty((768, ), device='cpu', dtype=torch.float32)
    buf958 = empty((768, ), device='cpu', dtype=torch.float32)
    buf959 = reinterpret_tensor(buf934, (1, 1024, 768), (786432, 768, 1), 0); del buf934  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_123(c_void_p(buf950.data_ptr()), c_void_p(buf943.data_ptr()), c_void_p(buf951.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(mul_1.data_ptr()), c_void_p(div_154.data_ptr()), c_void_p(getitem_3.data_ptr()), c_void_p(buf953.data_ptr()), c_void_p(buf954.data_ptr()), c_void_p(buf955.data_ptr()), c_void_p(buf956.data_ptr()), c_void_p(buf957.data_ptr()), c_void_p(buf958.data_ptr()), c_void_p(buf959.data_ptr()))
    del buf950
    del buf954
    del buf955
    del div_154
    del getitem_3
    del mul_1
    del primals_9
    buf960 = buf951; del buf951  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf959, (1024, 768), (768, 1), 0), permute_1402, out=buf960)
    del permute_1402
    buf961 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf959, (768, 1024), (1, 768), 0), view_69, out=buf961)
    del view_69
    buf962 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf963 = reinterpret_tensor(buf943, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf943  # reuse
    cpp_fused_clone_sum_124(c_void_p(buf959.data_ptr()), c_void_p(buf960.data_ptr()), c_void_p(buf962.data_ptr()), c_void_p(buf963.data_ptr()))
    buf964 = empty((48, 768, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1410, reinterpret_tensor(buf963, (48, 256, 64), (16384, 64, 1), 0), out=buf964)
    del permute_1410
    buf965 = reinterpret_tensor(buf914, (48, 256, 768), (196608, 768, 1), 0); del buf914  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf963, (48, 256, 64), (16384, 64, 1), 0), permute_1411, out=buf965)
    del permute_1411
    buf966 = buf881; del buf881  # reuse
    cpp_fused_new_zeros_125(c_void_p(buf966.data_ptr()))
    aten.index_put_(buf966, [reinterpret_tensor(buf27, (2359296, ), (1, ), 0)], reinterpret_tensor(buf964, (2359296, ), (1, ), 0), True)
    del buf27
    del buf964
    buf969 = buf884; del buf884  # reuse
    buf970 = reinterpret_tensor(buf910, (1024, 12, 513), (513, 525312, 1), 0); del buf910  # reuse
    buf971 = reinterpret_tensor(buf907, (6303744, ), (1, ), 0); del buf907  # reuse
    buf974 = reinterpret_tensor(buf904, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf904  # reuse
    buf975 = buf902; del buf902  # reuse
    buf978 = buf900; del buf900  # reuse
    buf980 = reinterpret_tensor(buf896, (1024, 12, 513), (513, 525312, 1), 0); del buf896  # reuse
    buf981 = reinterpret_tensor(buf895, (6303744, ), (1, ), 0); del buf895  # reuse
    buf983 = buf893; del buf893  # reuse
    buf985 = buf890; del buf890  # reuse
    buf987 = buf983; del buf983  # reuse
    buf989 = reinterpret_tensor(buf889, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf889  # reuse
    buf992 = reinterpret_tensor(buf886, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf886  # reuse
    buf995 = reinterpret_tensor(buf885, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf885  # reuse
    buf997 = buf912; del buf912  # reuse
    buf999 = reinterpret_tensor(buf880, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf880  # reuse
    cpp_fused__softmax_backward_data_add_as_strided_scatter_clone_constant_pad_nd_copy_masked_fill_native_dropout_backward_select_backward_slice_backward_tril_zeros_like_126(c_void_p(buf987.data_ptr()), c_void_p(unsqueeze_16.data_ptr()), c_void_p(buf965.data_ptr()), c_void_p(getitem_1.data_ptr()), c_void_p(alias_23.data_ptr()), c_void_p(rev_1.data_ptr()), c_void_p(slice_64.data_ptr()), c_void_p(buf969.data_ptr()), c_void_p(buf970.data_ptr()), c_void_p(buf971.data_ptr()), c_void_p(buf974.data_ptr()), c_void_p(buf975.data_ptr()), c_void_p(buf978.data_ptr()), c_void_p(buf980.data_ptr()), c_void_p(buf981.data_ptr()), c_void_p(buf985.data_ptr()), c_void_p(buf989.data_ptr()), c_void_p(buf992.data_ptr()), c_void_p(buf995.data_ptr()), c_void_p(buf997.data_ptr()), c_void_p(buf999.data_ptr()))
    del alias_23
    del buf965
    del buf969
    del buf970
    del buf971
    del buf974
    del buf975
    del buf978
    del buf980
    del buf981
    del buf985
    del buf987
    del buf989
    del buf992
    del buf995
    del buf997
    del getitem_1
    del rev_1
    del slice_64
    del unsqueeze_16
    buf1000 = reinterpret_tensor(buf916, (36, 64, 512), (32768, 512, 1), 0); del buf916  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_1421, reinterpret_tensor(buf999, (36, 512, 512), (262144, 512, 1), 0), out=buf1000)
    del permute_1421
    buf1001 = reinterpret_tensor(buf918, (36, 512, 64), (32768, 64, 1), 0); del buf918  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf999, (36, 512, 512), (262144, 512, 1), 0), permute_1422, out=buf1001)
    del buf999
    del permute_1422
    buf1002 = reinterpret_tensor(buf963, (786432, ), (1, ), 0); del buf963  # reuse
    buf1003 = reinterpret_tensor(buf915, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf915  # reuse
    cpp_fused_clone_index_add_127(c_void_p(buf1000.data_ptr()), c_void_p(buf1002.data_ptr()), c_void_p(buf1003.data_ptr()))
    del buf1000
    aten.index_put_(buf1002, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf1003, (1179648, ), (1, ), 0), True)
    del buf1003
    buf1006 = reinterpret_tensor(buf960, (786432, ), (1, ), 0); del buf960  # reuse
    cpp_fused_index_add_128(c_void_p(buf1006.data_ptr()))
    aten.index_put_(buf1006, [reinterpret_tensor(buf68, (1179648, ), (1, ), 0)], reinterpret_tensor(buf1001, (1179648, ), (1, ), 0), True)
    del buf1001
    del buf68
    buf1009 = reinterpret_tensor(buf959, (786432, ), (1, ), 0); del buf959  # reuse
    buf1012 = reinterpret_tensor(buf931, (1024, 1, 768), (768, 768, 1), 0); del buf931  # reuse
    buf1013 = buf928; del buf928  # reuse
    buf1015 = reinterpret_tensor(buf871, (1024, 768), (768, 1), 0); del buf871  # reuse
    cpp_fused_as_strided_scatter_div_view_129(c_void_p(buf1006.data_ptr()), c_void_p(buf966.data_ptr()), c_void_p(buf1009.data_ptr()), c_void_p(buf1012.data_ptr()), c_void_p(buf1013.data_ptr()), c_void_p(buf1015.data_ptr()))
    del buf1006
    del buf1009
    del buf966
    buf1016 = reinterpret_tensor(buf1012, (1024, 768), (768, 1), 0); del buf1012  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf1015, permute_1433, out=buf1016)
    del permute_1433
    buf1017 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf1015, (768, 1024), (1, 768), 0), view, out=buf1017)
    buf1018 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_130(c_void_p(buf1015.data_ptr()), c_void_p(buf1018.data_ptr()))
    buf1019 = buf1015; del buf1015  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf1002, (1024, 768), (768, 1), 0), permute_1437, out=buf1019)
    del permute_1437
    buf1020 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf1002, (768, 1024), (1, 768), 0), view, out=buf1020)
    buf1021 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_131(c_void_p(buf1002.data_ptr()), c_void_p(buf1021.data_ptr()))
    buf1022 = reinterpret_tensor(buf1002, (1024, 768), (768, 1), 0); del buf1002  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf1013, permute_1446, out=buf1022)
    del permute_1446
    buf1023 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf1013, (768, 1024), (1, 768), 0), view, out=buf1023)
    del view
    buf1024 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf1025 = reinterpret_tensor(buf1016, (1, 1024, 768), (786432, 768, 1), 0); del buf1016  # reuse
    cpp_fused_add_sum_132(c_void_p(buf1025.data_ptr()), c_void_p(buf1013.data_ptr()), c_void_p(buf956.data_ptr()), c_void_p(buf1019.data_ptr()), c_void_p(buf1022.data_ptr()), c_void_p(buf1024.data_ptr()))
    return (reinterpret_tensor(buf1023, (768, 768), (768, 1), 0), reinterpret_tensor(buf1024, (768, ), (1, ), 0), reinterpret_tensor(buf1020, (768, 768), (768, 1), 0), reinterpret_tensor(buf1021, (768, ), (1, ), 0), reinterpret_tensor(buf1017, (768, 768), (768, 1), 0), reinterpret_tensor(buf1018, (768, ), (1, ), 0), reinterpret_tensor(buf961, (768, 768), (768, 1), 0), reinterpret_tensor(buf962, (768, ), (1, ), 0), buf957, buf958, reinterpret_tensor(buf952, (3072, 768), (768, 1), 0), reinterpret_tensor(buf953, (3072, ), (1, ), 0), reinterpret_tensor(buf948, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf949, (768, ), (1, ), 0), buf944, buf945, reinterpret_tensor(buf938, (768, 768), (768, 1), 0), reinterpret_tensor(buf939, (768, ), (1, ), 0), reinterpret_tensor(buf935, (768, 768), (768, 1), 0), reinterpret_tensor(buf936, (768, ), (1, ), 0), reinterpret_tensor(buf932, (768, 768), (768, 1), 0), reinterpret_tensor(buf933, (768, ), (1, ), 0), reinterpret_tensor(buf876, (768, 768), (768, 1), 0), reinterpret_tensor(buf877, (768, ), (1, ), 0), buf872, buf873, reinterpret_tensor(buf867, (3072, 768), (768, 1), 0), reinterpret_tensor(buf868, (3072, ), (1, ), 0), reinterpret_tensor(buf863, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf864, (768, ), (1, ), 0), buf859, buf860, reinterpret_tensor(buf853, (768, 768), (768, 1), 0), reinterpret_tensor(buf854, (768, ), (1, ), 0), reinterpret_tensor(buf850, (768, 768), (768, 1), 0), reinterpret_tensor(buf851, (768, ), (1, ), 0), reinterpret_tensor(buf847, (768, 768), (768, 1), 0), reinterpret_tensor(buf848, (768, ), (1, ), 0), reinterpret_tensor(buf791, (768, 768), (768, 1), 0), reinterpret_tensor(buf792, (768, ), (1, ), 0), buf787, buf788, reinterpret_tensor(buf782, (3072, 768), (768, 1), 0), reinterpret_tensor(buf783, (3072, ), (1, ), 0), reinterpret_tensor(buf778, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf779, (768, ), (1, ), 0), buf774, buf775, reinterpret_tensor(buf768, (768, 768), (768, 1), 0), reinterpret_tensor(buf769, (768, ), (1, ), 0), reinterpret_tensor(buf765, (768, 768), (768, 1), 0), reinterpret_tensor(buf766, (768, ), (1, ), 0), reinterpret_tensor(buf762, (768, 768), (768, 1), 0), reinterpret_tensor(buf763, (768, ), (1, ), 0), reinterpret_tensor(buf706, (768, 768), (768, 1), 0), reinterpret_tensor(buf707, (768, ), (1, ), 0), buf702, buf703, reinterpret_tensor(buf697, (3072, 768), (768, 1), 0), reinterpret_tensor(buf698, (3072, ), (1, ), 0), reinterpret_tensor(buf693, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf694, (768, ), (1, ), 0), buf689, buf690, reinterpret_tensor(buf683, (768, 768), (768, 1), 0), reinterpret_tensor(buf684, (768, ), (1, ), 0), reinterpret_tensor(buf680, (768, 768), (768, 1), 0), reinterpret_tensor(buf681, (768, ), (1, ), 0), reinterpret_tensor(buf677, (768, 768), (768, 1), 0), reinterpret_tensor(buf678, (768, ), (1, ), 0), reinterpret_tensor(buf621, (768, 768), (768, 1), 0), reinterpret_tensor(buf622, (768, ), (1, ), 0), buf617, buf618, reinterpret_tensor(buf612, (3072, 768), (768, 1), 0), reinterpret_tensor(buf613, (3072, ), (1, ), 0), reinterpret_tensor(buf608, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf609, (768, ), (1, ), 0), buf604, buf605, reinterpret_tensor(buf598, (768, 768), (768, 1), 0), reinterpret_tensor(buf599, (768, ), (1, ), 0), reinterpret_tensor(buf595, (768, 768), (768, 1), 0), reinterpret_tensor(buf596, (768, ), (1, ), 0), reinterpret_tensor(buf592, (768, 768), (768, 1), 0), reinterpret_tensor(buf593, (768, ), (1, ), 0), reinterpret_tensor(buf536, (768, 768), (768, 1), 0), reinterpret_tensor(buf537, (768, ), (1, ), 0), buf532, buf533, reinterpret_tensor(buf527, (3072, 768), (768, 1), 0), reinterpret_tensor(buf528, (3072, ), (1, ), 0), reinterpret_tensor(buf523, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf524, (768, ), (1, ), 0), buf519, buf520, reinterpret_tensor(buf513, (768, 768), (768, 1), 0), reinterpret_tensor(buf514, (768, ), (1, ), 0), reinterpret_tensor(buf510, (768, 768), (768, 1), 0), reinterpret_tensor(buf511, (768, ), (1, ), 0), reinterpret_tensor(buf507, (768, 768), (768, 1), 0), reinterpret_tensor(buf508, (768, ), (1, ), 0), reinterpret_tensor(buf451, (768, 768), (768, 1), 0), reinterpret_tensor(buf452, (768, ), (1, ), 0), buf447, buf448, reinterpret_tensor(buf442, (3072, 768), (768, 1), 0), reinterpret_tensor(buf443, (3072, ), (1, ), 0), reinterpret_tensor(buf438, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf439, (768, ), (1, ), 0), buf434, buf435, reinterpret_tensor(buf428, (768, 768), (768, 1), 0), reinterpret_tensor(buf429, (768, ), (1, ), 0), reinterpret_tensor(buf425, (768, 768), (768, 1), 0), reinterpret_tensor(buf426, (768, ), (1, ), 0), reinterpret_tensor(buf422, (768, 768), (768, 1), 0), reinterpret_tensor(buf423, (768, ), (1, ), 0), reinterpret_tensor(buf366, (768, 768), (768, 1), 0), reinterpret_tensor(buf367, (768, ), (1, ), 0), buf362, buf363, reinterpret_tensor(buf357, (3072, 768), (768, 1), 0), reinterpret_tensor(buf358, (3072, ), (1, ), 0), reinterpret_tensor(buf353, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf354, (768, ), (1, ), 0), buf349, buf350, reinterpret_tensor(buf343, (768, 768), (768, 1), 0), reinterpret_tensor(buf344, (768, ), (1, ), 0), reinterpret_tensor(buf340, (768, 768), (768, 1), 0), reinterpret_tensor(buf341, (768, ), (1, ), 0), reinterpret_tensor(buf337, (768, 768), (768, 1), 0), reinterpret_tensor(buf338, (768, ), (1, ), 0), reinterpret_tensor(buf281, (768, 768), (768, 1), 0), reinterpret_tensor(buf282, (768, ), (1, ), 0), buf277, buf278, reinterpret_tensor(buf272, (3072, 768), (768, 1), 0), reinterpret_tensor(buf273, (3072, ), (1, ), 0), reinterpret_tensor(buf268, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf269, (768, ), (1, ), 0), buf264, buf265, reinterpret_tensor(buf258, (768, 768), (768, 1), 0), reinterpret_tensor(buf259, (768, ), (1, ), 0), reinterpret_tensor(buf255, (768, 768), (768, 1), 0), reinterpret_tensor(buf256, (768, ), (1, ), 0), reinterpret_tensor(buf252, (768, 768), (768, 1), 0), reinterpret_tensor(buf253, (768, ), (1, ), 0), reinterpret_tensor(buf196, (768, 768), (768, 1), 0), reinterpret_tensor(buf197, (768, ), (1, ), 0), buf192, buf193, reinterpret_tensor(buf187, (3072, 768), (768, 1), 0), reinterpret_tensor(buf188, (3072, ), (1, ), 0), reinterpret_tensor(buf183, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf184, (768, ), (1, ), 0), buf179, buf180, reinterpret_tensor(buf173, (768, 768), (768, 1), 0), reinterpret_tensor(buf174, (768, ), (1, ), 0), reinterpret_tensor(buf170, (768, 768), (768, 1), 0), reinterpret_tensor(buf171, (768, ), (1, ), 0), reinterpret_tensor(buf167, (768, 768), (768, 1), 0), reinterpret_tensor(buf168, (768, ), (1, ), 0), reinterpret_tensor(buf111, (768, 768), (768, 1), 0), reinterpret_tensor(buf112, (768, ), (1, ), 0), buf107, buf108, reinterpret_tensor(buf102, (3072, 768), (768, 1), 0), reinterpret_tensor(buf103, (3072, ), (1, ), 0), reinterpret_tensor(buf98, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf99, (768, ), (1, ), 0), buf94, buf95, reinterpret_tensor(buf88, (768, 768), (768, 1), 0), reinterpret_tensor(buf89, (768, ), (1, ), 0), reinterpret_tensor(buf85, (768, 768), (768, 1), 0), reinterpret_tensor(buf86, (768, ), (1, ), 0), reinterpret_tensor(buf82, (768, 768), (768, 1), 0), reinterpret_tensor(buf83, (768, ), (1, ), 0), reinterpret_tensor(buf20, (768, 768), (768, 1), 0), reinterpret_tensor(buf21, (768, ), (1, ), 0), buf16, buf17, reinterpret_tensor(buf11, (3072, 768), (768, 1), 0), reinterpret_tensor(buf12, (3072, ), (1, ), 0), reinterpret_tensor(buf7, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf8, (768, ), (1, ), 0), buf3, buf4, buf1025, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_9 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    view = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    slice_64 = rand_strided((1, 256, 1, 257), (65792, 257, 257, 1), device='cpu', dtype=torch.float32)
    rev_1 = rand_strided((1, 256, 1, 257), (65792, 257, 257, 1), device='cpu', dtype=torch.float32)
    unsqueeze_16 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.bool)
    getitem_1 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.bool)
    view_69 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_1 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_71 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_4 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_73 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_6 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_75 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_11 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.bool)
    view_144 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_13 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_9 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_146 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_148 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_17 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_14 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_150 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_21 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.bool)
    view_219 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_23 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_17 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_221 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_16 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_223 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_27 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_22 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_225 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_31 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.bool)
    view_294 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_33 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_25 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_296 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_298 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_37 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_30 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_300 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_41 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.bool)
    view_369 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_43 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_33 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_371 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_28 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_373 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_47 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_38 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_375 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_51 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.bool)
    view_444 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_53 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_41 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_446 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_448 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_57 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_46 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_450 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_61 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.bool)
    view_519 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_63 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_49 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_521 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_40 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_523 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_67 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_54 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_525 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_71 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.bool)
    view_594 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_73 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_57 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_596 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_46 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_598 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_77 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_62 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_600 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_81 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.bool)
    view_669 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_83 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_65 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_671 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_52 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_673 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_87 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_70 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_675 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_91 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.bool)
    view_744 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_93 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_73 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_746 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_58 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_748 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_97 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_78 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_750 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_101 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.bool)
    view_819 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_103 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_81 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_821 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_64 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_823 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_107 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_86 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_825 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_111 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.bool)
    view_894 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_113 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_89 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_896 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_70 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_898 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_117 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_94 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    div_120 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_756 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_760 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_121 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_764 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_772 = rand_strided((48, 768, 256), (197120, 1, 769), device='cpu', dtype=torch.float32)
    permute_773 = rand_strided((48, 64, 768), (49152, 1, 64), device='cpu', dtype=torch.float32)
    alias_12 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.float32)
    permute_783 = rand_strided((36, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_784 = rand_strided((36, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_795 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_799 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_808 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_123 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_814 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_818 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_124 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_822 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_830 = rand_strided((48, 768, 256), (197120, 1, 769), device='cpu', dtype=torch.float32)
    permute_831 = rand_strided((48, 64, 768), (49152, 1, 64), device='cpu', dtype=torch.float32)
    alias_13 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.float32)
    permute_841 = rand_strided((36, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_842 = rand_strided((36, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_853 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_857 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_866 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_126 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_872 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_876 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_127 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_880 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_888 = rand_strided((48, 768, 256), (197120, 1, 769), device='cpu', dtype=torch.float32)
    permute_889 = rand_strided((48, 64, 768), (49152, 1, 64), device='cpu', dtype=torch.float32)
    alias_14 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.float32)
    permute_899 = rand_strided((36, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_900 = rand_strided((36, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_911 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_915 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_924 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_129 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_930 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_934 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_130 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_938 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_946 = rand_strided((48, 768, 256), (197120, 1, 769), device='cpu', dtype=torch.float32)
    permute_947 = rand_strided((48, 64, 768), (49152, 1, 64), device='cpu', dtype=torch.float32)
    alias_15 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.float32)
    permute_957 = rand_strided((36, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_958 = rand_strided((36, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_969 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_973 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_982 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_132 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_988 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_992 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_133 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_996 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1004 = rand_strided((48, 768, 256), (197120, 1, 769), device='cpu', dtype=torch.float32)
    permute_1005 = rand_strided((48, 64, 768), (49152, 1, 64), device='cpu', dtype=torch.float32)
    alias_16 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.float32)
    permute_1015 = rand_strided((36, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_1016 = rand_strided((36, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_1027 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1031 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1040 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_135 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_1046 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_1050 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_136 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_1054 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1062 = rand_strided((48, 768, 256), (197120, 1, 769), device='cpu', dtype=torch.float32)
    permute_1063 = rand_strided((48, 64, 768), (49152, 1, 64), device='cpu', dtype=torch.float32)
    alias_17 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.float32)
    permute_1073 = rand_strided((36, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_1074 = rand_strided((36, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_1085 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1089 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1098 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_138 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_1104 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_1108 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_139 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_1112 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1120 = rand_strided((48, 768, 256), (197120, 1, 769), device='cpu', dtype=torch.float32)
    permute_1121 = rand_strided((48, 64, 768), (49152, 1, 64), device='cpu', dtype=torch.float32)
    alias_18 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.float32)
    permute_1131 = rand_strided((36, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_1132 = rand_strided((36, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_1143 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1147 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1156 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_141 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_1162 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_1166 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_142 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_1170 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1178 = rand_strided((48, 768, 256), (197120, 1, 769), device='cpu', dtype=torch.float32)
    permute_1179 = rand_strided((48, 64, 768), (49152, 1, 64), device='cpu', dtype=torch.float32)
    alias_19 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.float32)
    permute_1189 = rand_strided((36, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_1190 = rand_strided((36, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_1201 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1205 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1214 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_144 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_1220 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_1224 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_145 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_1228 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1236 = rand_strided((48, 768, 256), (197120, 1, 769), device='cpu', dtype=torch.float32)
    permute_1237 = rand_strided((48, 64, 768), (49152, 1, 64), device='cpu', dtype=torch.float32)
    alias_20 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.float32)
    permute_1247 = rand_strided((36, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_1248 = rand_strided((36, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_1259 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1263 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1272 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_147 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_1278 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_1282 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_148 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_1286 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1294 = rand_strided((48, 768, 256), (197120, 1, 769), device='cpu', dtype=torch.float32)
    permute_1295 = rand_strided((48, 64, 768), (49152, 1, 64), device='cpu', dtype=torch.float32)
    alias_21 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.float32)
    permute_1305 = rand_strided((36, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_1306 = rand_strided((36, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_1317 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1321 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1330 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_150 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_1336 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_1340 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_151 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_1344 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1352 = rand_strided((48, 768, 256), (197120, 1, 769), device='cpu', dtype=torch.float32)
    permute_1353 = rand_strided((48, 64, 768), (49152, 1, 64), device='cpu', dtype=torch.float32)
    alias_22 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.float32)
    permute_1363 = rand_strided((36, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_1364 = rand_strided((36, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_1375 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1379 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1388 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_153 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_1394 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_1398 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_154 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_1402 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1410 = rand_strided((48, 768, 256), (197120, 1, 769), device='cpu', dtype=torch.float32)
    permute_1411 = rand_strided((48, 64, 768), (49152, 1, 64), device='cpu', dtype=torch.float32)
    alias_23 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cpu', dtype=torch.float32)
    permute_1421 = rand_strided((36, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_1422 = rand_strided((36, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_1433 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1437 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1446 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_9, primals_15, primals_25, primals_31, primals_41, primals_47, primals_57, primals_63, primals_73, primals_79, primals_89, primals_95, primals_105, primals_111, primals_121, primals_127, primals_137, primals_143, primals_153, primals_159, primals_169, primals_175, primals_185, primals_191, view, slice_64, rev_1, unsqueeze_16, getitem_1, view_69, getitem_3, mul_1, view_71, addmm_4, view_73, getitem_7, mul_6, view_75, getitem_11, view_144, getitem_13, mul_9, view_146, addmm_10, view_148, getitem_17, mul_14, view_150, getitem_21, view_219, getitem_23, mul_17, view_221, addmm_16, view_223, getitem_27, mul_22, view_225, getitem_31, view_294, getitem_33, mul_25, view_296, addmm_22, view_298, getitem_37, mul_30, view_300, getitem_41, view_369, getitem_43, mul_33, view_371, addmm_28, view_373, getitem_47, mul_38, view_375, getitem_51, view_444, getitem_53, mul_41, view_446, addmm_34, view_448, getitem_57, mul_46, view_450, getitem_61, view_519, getitem_63, mul_49, view_521, addmm_40, view_523, getitem_67, mul_54, view_525, getitem_71, view_594, getitem_73, mul_57, view_596, addmm_46, view_598, getitem_77, mul_62, view_600, getitem_81, view_669, getitem_83, mul_65, view_671, addmm_52, view_673, getitem_87, mul_70, view_675, getitem_91, view_744, getitem_93, mul_73, view_746, addmm_58, view_748, getitem_97, mul_78, view_750, getitem_101, view_819, getitem_103, mul_81, view_821, addmm_64, view_823, getitem_107, mul_86, view_825, getitem_111, view_894, getitem_113, mul_89, view_896, addmm_70, view_898, getitem_117, mul_94, div_120, permute_756, permute_760, div_121, permute_764, permute_772, permute_773, alias_12, permute_783, permute_784, permute_795, permute_799, permute_808, div_123, permute_814, permute_818, div_124, permute_822, permute_830, permute_831, alias_13, permute_841, permute_842, permute_853, permute_857, permute_866, div_126, permute_872, permute_876, div_127, permute_880, permute_888, permute_889, alias_14, permute_899, permute_900, permute_911, permute_915, permute_924, div_129, permute_930, permute_934, div_130, permute_938, permute_946, permute_947, alias_15, permute_957, permute_958, permute_969, permute_973, permute_982, div_132, permute_988, permute_992, div_133, permute_996, permute_1004, permute_1005, alias_16, permute_1015, permute_1016, permute_1027, permute_1031, permute_1040, div_135, permute_1046, permute_1050, div_136, permute_1054, permute_1062, permute_1063, alias_17, permute_1073, permute_1074, permute_1085, permute_1089, permute_1098, div_138, permute_1104, permute_1108, div_139, permute_1112, permute_1120, permute_1121, alias_18, permute_1131, permute_1132, permute_1143, permute_1147, permute_1156, div_141, permute_1162, permute_1166, div_142, permute_1170, permute_1178, permute_1179, alias_19, permute_1189, permute_1190, permute_1201, permute_1205, permute_1214, div_144, permute_1220, permute_1224, div_145, permute_1228, permute_1236, permute_1237, alias_20, permute_1247, permute_1248, permute_1259, permute_1263, permute_1272, div_147, permute_1278, permute_1282, div_148, permute_1286, permute_1294, permute_1295, alias_21, permute_1305, permute_1306, permute_1317, permute_1321, permute_1330, div_150, permute_1336, permute_1340, div_151, permute_1344, permute_1352, permute_1353, alias_22, permute_1363, permute_1364, permute_1375, permute_1379, permute_1388, div_153, permute_1394, permute_1398, div_154, permute_1402, permute_1410, permute_1411, alias_23, permute_1421, permute_1422, permute_1433, permute_1437, permute_1446, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AllenaiLongformerBase', benchmark_compiled_module)
