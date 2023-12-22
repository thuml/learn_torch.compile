
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


cpp_fused_new_zeros_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
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
                    auto tmp0 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = out_ptr1[static_cast<long>(x0)];
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
                    tmp18.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
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
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_add_mul_pow_sum_tanh_backward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                       float* out_ptr5)
{
    auto in_ptr1 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr6[static_cast<long>(x0)];
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


cpp_fused_sum_4 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (1024L*x1))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp3 = c10::convert<float>(tmp2);
                        auto tmp4 = static_cast<float>(1.1111111111111112);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp0 ? tmp11 : tmp12;
                        auto tmp14 = static_cast<float>(8.0);
                        auto tmp15 = tmp13 / tmp14;
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp14 = in_ptr2[static_cast<long>(x0 + (1024L*(static_cast<long>(x1) % static_cast<long>(768L))))];
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp17 = tmp0 >= tmp9;
                    auto tmp18 = static_cast<long>(2304);
                    auto tmp19 = tmp0 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp22 = in_ptr4[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp25 = tmp11 ? tmp16 : tmp24;
                    auto tmp26 = tmp4 ? tmp7 : tmp25;
                    out_ptr0[static_cast<long>(x1 + (2304L*x0))] = tmp26;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2304L*x1)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_add_mul_pow_sum_tanh_backward_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (1024L*x1))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp3 = c10::convert<float>(tmp2);
                        auto tmp4 = static_cast<float>(1.1111111111111112);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp0 ? tmp11 : tmp12;
                        auto tmp14 = static_cast<float>(8.0);
                        auto tmp15 = tmp13 / tmp14;
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp14 = in_ptr2[static_cast<long>(x0 + (1024L*(static_cast<long>(x1) % static_cast<long>(768L))))];
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp17 = tmp0 >= tmp9;
                    auto tmp18 = static_cast<long>(2304);
                    auto tmp19 = tmp0 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp22 = in_ptr4[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp25 = tmp11 ? tmp16 : tmp24;
                    auto tmp26 = tmp4 ? tmp7 : tmp25;
                    out_ptr0[static_cast<long>(x1 + (2304L*x0))] = tmp26;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2304L*x1)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_add_mul_pow_sum_tanh_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_sum_16 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (1024L*x1))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp3 = c10::convert<float>(tmp2);
                        auto tmp4 = static_cast<float>(1.1111111111111112);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp0 ? tmp11 : tmp12;
                        auto tmp14 = static_cast<float>(8.0);
                        auto tmp15 = tmp13 / tmp14;
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp14 = in_ptr2[static_cast<long>(x0 + (1024L*(static_cast<long>(x1) % static_cast<long>(768L))))];
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp17 = tmp0 >= tmp9;
                    auto tmp18 = static_cast<long>(2304);
                    auto tmp19 = tmp0 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp22 = in_ptr4[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp25 = tmp11 ? tmp16 : tmp24;
                    auto tmp26 = tmp4 ? tmp7 : tmp25;
                    out_ptr0[static_cast<long>(x1 + (2304L*x0))] = tmp26;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2304L*x1)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_add_mul_pow_sum_tanh_backward_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_sum_22 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (1024L*x1))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp3 = c10::convert<float>(tmp2);
                        auto tmp4 = static_cast<float>(1.1111111111111112);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp0 ? tmp11 : tmp12;
                        auto tmp14 = static_cast<float>(8.0);
                        auto tmp15 = tmp13 / tmp14;
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp14 = in_ptr2[static_cast<long>(x0 + (1024L*(static_cast<long>(x1) % static_cast<long>(768L))))];
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp17 = tmp0 >= tmp9;
                    auto tmp18 = static_cast<long>(2304);
                    auto tmp19 = tmp0 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp22 = in_ptr4[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp25 = tmp11 ? tmp16 : tmp24;
                    auto tmp26 = tmp4 ? tmp7 : tmp25;
                    out_ptr0[static_cast<long>(x1 + (2304L*x0))] = tmp26;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2304L*x1)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_add_mul_pow_sum_tanh_backward_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_sum_28 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (1024L*x1))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp3 = c10::convert<float>(tmp2);
                        auto tmp4 = static_cast<float>(1.1111111111111112);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp0 ? tmp11 : tmp12;
                        auto tmp14 = static_cast<float>(8.0);
                        auto tmp15 = tmp13 / tmp14;
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp14 = in_ptr2[static_cast<long>(x0 + (1024L*(static_cast<long>(x1) % static_cast<long>(768L))))];
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp17 = tmp0 >= tmp9;
                    auto tmp18 = static_cast<long>(2304);
                    auto tmp19 = tmp0 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp22 = in_ptr4[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp25 = tmp11 ? tmp16 : tmp24;
                    auto tmp26 = tmp4 ? tmp7 : tmp25;
                    out_ptr0[static_cast<long>(x1 + (2304L*x0))] = tmp26;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2304L*x1)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_add_mul_pow_sum_tanh_backward_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
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
                       const bool* in_ptr5,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_sum_34 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (1024L*x1))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp3 = c10::convert<float>(tmp2);
                        auto tmp4 = static_cast<float>(1.1111111111111112);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp0 ? tmp11 : tmp12;
                        auto tmp14 = static_cast<float>(8.0);
                        auto tmp15 = tmp13 / tmp14;
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp14 = in_ptr2[static_cast<long>(x0 + (1024L*(static_cast<long>(x1) % static_cast<long>(768L))))];
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp17 = tmp0 >= tmp9;
                    auto tmp18 = static_cast<long>(2304);
                    auto tmp19 = tmp0 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp22 = in_ptr4[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp25 = tmp11 ? tmp16 : tmp24;
                    auto tmp26 = tmp4 ? tmp7 : tmp25;
                    out_ptr0[static_cast<long>(x1 + (2304L*x0))] = tmp26;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2304L*x1)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_add_mul_pow_sum_tanh_backward_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_sum_40 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (1024L*x1))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp3 = c10::convert<float>(tmp2);
                        auto tmp4 = static_cast<float>(1.1111111111111112);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp0 ? tmp11 : tmp12;
                        auto tmp14 = static_cast<float>(8.0);
                        auto tmp15 = tmp13 / tmp14;
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp14 = in_ptr2[static_cast<long>(x0 + (1024L*(static_cast<long>(x1) % static_cast<long>(768L))))];
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp17 = tmp0 >= tmp9;
                    auto tmp18 = static_cast<long>(2304);
                    auto tmp19 = tmp0 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp22 = in_ptr4[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp25 = tmp11 ? tmp16 : tmp24;
                    auto tmp26 = tmp4 ? tmp7 : tmp25;
                    out_ptr0[static_cast<long>(x1 + (2304L*x0))] = tmp26;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2304L*x1)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_add_mul_pow_sum_tanh_backward_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_sum_46 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (1024L*x1))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp3 = c10::convert<float>(tmp2);
                        auto tmp4 = static_cast<float>(1.1111111111111112);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp0 ? tmp11 : tmp12;
                        auto tmp14 = static_cast<float>(8.0);
                        auto tmp15 = tmp13 / tmp14;
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp14 = in_ptr2[static_cast<long>(x0 + (1024L*(static_cast<long>(x1) % static_cast<long>(768L))))];
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp17 = tmp0 >= tmp9;
                    auto tmp18 = static_cast<long>(2304);
                    auto tmp19 = tmp0 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp22 = in_ptr4[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp25 = tmp11 ? tmp16 : tmp24;
                    auto tmp26 = tmp4 ? tmp7 : tmp25;
                    out_ptr0[static_cast<long>(x1 + (2304L*x0))] = tmp26;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2304L*x1)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_add_mul_pow_sum_tanh_backward_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_sum_52 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (1024L*x1))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp3 = c10::convert<float>(tmp2);
                        auto tmp4 = static_cast<float>(1.1111111111111112);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp0 ? tmp11 : tmp12;
                        auto tmp14 = static_cast<float>(8.0);
                        auto tmp15 = tmp13 / tmp14;
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp14 = in_ptr2[static_cast<long>(x0 + (1024L*(static_cast<long>(x1) % static_cast<long>(768L))))];
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp17 = tmp0 >= tmp9;
                    auto tmp18 = static_cast<long>(2304);
                    auto tmp19 = tmp0 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp22 = in_ptr4[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp25 = tmp11 ? tmp16 : tmp24;
                    auto tmp26 = tmp4 ? tmp7 : tmp25;
                    out_ptr0[static_cast<long>(x1 + (2304L*x0))] = tmp26;
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
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2304L*x1)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_add_mul_pow_sum_tanh_backward_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_sum_58 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (1024L*x1))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp3 = c10::convert<float>(tmp2);
                        auto tmp4 = static_cast<float>(1.1111111111111112);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp0 ? tmp11 : tmp12;
                        auto tmp14 = static_cast<float>(8.0);
                        auto tmp15 = tmp13 / tmp14;
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_60 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp14 = in_ptr2[static_cast<long>(x0 + (1024L*(static_cast<long>(x1) % static_cast<long>(768L))))];
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp17 = tmp0 >= tmp9;
                    auto tmp18 = static_cast<long>(2304);
                    auto tmp19 = tmp0 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp22 = in_ptr4[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp25 = tmp11 ? tmp16 : tmp24;
                    auto tmp26 = tmp4 ? tmp7 : tmp25;
                    out_ptr0[static_cast<long>(x1 + (2304L*x0))] = tmp26;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2304L*x1)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_add_mul_pow_sum_tanh_backward_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (1024L*x1))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp3 = c10::convert<float>(tmp2);
                        auto tmp4 = static_cast<float>(1.1111111111111112);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp0 ? tmp11 : tmp12;
                        auto tmp14 = static_cast<float>(8.0);
                        auto tmp15 = tmp13 / tmp14;
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp14 = in_ptr2[static_cast<long>(x0 + (1024L*(static_cast<long>(x1) % static_cast<long>(768L))))];
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp17 = tmp0 >= tmp9;
                    auto tmp18 = static_cast<long>(2304);
                    auto tmp19 = tmp0 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp22 = in_ptr4[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp25 = tmp11 ? tmp16 : tmp24;
                    auto tmp26 = tmp4 ? tmp7 : tmp25;
                    out_ptr0[static_cast<long>(x1 + (2304L*x0))] = tmp26;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2304L*x1)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_add_mul_pow_sum_tanh_backward_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 * tmp3;
                auto tmp5 = tmp0 * tmp4;
                auto tmp7 = tmp6 * tmp6;
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp9 - tmp7;
                auto tmp11 = tmp5 * tmp10;
                auto tmp12 = static_cast<float>(0.7978845608028654);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.044715);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                auto tmp18 = tmp1 * tmp1;
                auto tmp19 = static_cast<float>(3.0);
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 * tmp20;
                auto tmp22 = tmp17 * tmp21;
                auto tmp23 = tmp14 + tmp22;
                auto tmp24 = tmp6 + tmp9;
                auto tmp25 = tmp0 * tmp24;
                auto tmp26 = tmp25 * tmp3;
                auto tmp27 = tmp23 + tmp26;
                tmp27.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp10 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp16 * tmp15;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_sum_70 = async_compile.cpp('''
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


cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = c10::convert<float>(tmp1);
                        auto tmp3 = static_cast<float>(1.1111111111111112);
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                        auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (1024L*x1))];
                        auto tmp1 = in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp2 = in_ptr1[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))];
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp3 = c10::convert<float>(tmp2);
                        auto tmp4 = static_cast<float>(1.1111111111111112);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                        auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp0 ? tmp11 : tmp12;
                        auto tmp14 = static_cast<float>(8.0);
                        auto tmp15 = tmp13 / tmp14;
                        in_out_ptr0[static_cast<long>(x2 + (1024L*x1) + (1048576L*x0))] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_72 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2304L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(768);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(1536);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp14 = in_ptr2[static_cast<long>(x0 + (1024L*(static_cast<long>(x1) % static_cast<long>(768L))))];
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp17 = tmp0 >= tmp9;
                    auto tmp18 = static_cast<long>(2304);
                    auto tmp19 = tmp0 < tmp18;
                    auto tmp20 = [&]
                    {
                        auto tmp21 = in_ptr3[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp22 = in_ptr4[static_cast<long>((64L*x0) + (65536L*(static_cast<long>(c10::div_floor_integer(x1, 64L)) % static_cast<long>(12L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                    auto tmp25 = tmp11 ? tmp16 : tmp24;
                    auto tmp26 = tmp4 ? tmp7 : tmp25;
                    out_ptr0[static_cast<long>(x1 + (2304L*x0))] = tmp26;
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_scalar_tensor_sum_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       const long* in_ptr6,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2304L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2304L*x1)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (768L*x0))];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                    auto tmp3 = in_ptr2[static_cast<long>(x1)];
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp9 = in_ptr3[static_cast<long>(x1 + (768L*x0))];
                    auto tmp10 = out_ptr2[static_cast<long>(x0)];
                    auto tmp15 = in_ptr5[static_cast<long>(x1 + (768L*x0))];
                    auto tmp20 = in_ptr6[static_cast<long>(x0)];
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                    auto tmp13 = decltype(tmp1)(tmp1 * tmp12);
                    auto tmp14 = decltype(tmp0)(tmp0 + tmp13);
                    auto tmp16 = c10::convert<float>(tmp15);
                    auto tmp17 = static_cast<float>(1.1111111111111112);
                    auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                    auto tmp19 = decltype(tmp14)(tmp14 * tmp18);
                    auto tmp21 = static_cast<long>(-1);
                    auto tmp22 = tmp20 == tmp21;
                    auto tmp23 = static_cast<float>(0.0);
                    auto tmp24 = tmp22 ? tmp23 : tmp19;
                    in_out_ptr0[static_cast<long>(x1 + (768L*x0))] = tmp19;
                    out_ptr5[static_cast<long>(x1 + (768L*x0))] = tmp24;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr6 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<bool>(0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 ? tmp2 : tmp0;
                in_out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(38597376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, view, view_1, getitem_1, mul, getitem_8, getitem_10, mul_2, addmm_2, tanh, getitem_14, mul_8, getitem_21, getitem_23, mul_10, addmm_6, tanh_1, getitem_27, mul_16, getitem_34, getitem_36, mul_18, addmm_10, tanh_2, getitem_40, mul_24, getitem_47, getitem_49, mul_26, addmm_14, tanh_3, getitem_53, mul_32, getitem_60, getitem_62, mul_34, addmm_18, tanh_4, getitem_66, mul_40, getitem_73, getitem_75, mul_42, addmm_22, tanh_5, getitem_79, mul_48, getitem_86, getitem_88, mul_50, addmm_26, tanh_6, getitem_92, mul_56, getitem_99, getitem_101, mul_58, addmm_30, tanh_7, getitem_105, mul_64, getitem_112, getitem_114, mul_66, addmm_34, tanh_8, getitem_118, mul_72, getitem_125, getitem_127, mul_74, addmm_38, tanh_9, getitem_131, mul_80, getitem_138, getitem_140, mul_82, addmm_42, tanh_10, getitem_144, mul_88, getitem_151, getitem_153, mul_90, addmm_46, tanh_11, getitem_157, mul_96, view_219, sub_37, full_default_24, permute_63, div_24, permute_65, permute_66, permute_67, permute_68, div_25, permute_69, permute_70, permute_72, permute_73, alias_25, permute_74, permute_75, permute_80, permute_81, div_27, permute_82, permute_83, permute_84, permute_85, div_28, permute_86, permute_87, permute_89, permute_90, alias_27, permute_91, permute_92, permute_97, permute_98, div_30, permute_99, permute_100, permute_101, permute_102, div_31, permute_103, permute_104, permute_106, permute_107, alias_29, permute_108, permute_109, permute_114, permute_115, div_33, permute_116, permute_117, permute_118, permute_119, div_34, permute_120, permute_121, permute_123, permute_124, alias_31, permute_125, permute_126, permute_131, permute_132, div_36, permute_133, permute_134, permute_135, permute_136, div_37, permute_137, permute_138, permute_140, permute_141, alias_33, permute_142, permute_143, permute_148, permute_149, div_39, permute_150, permute_151, permute_152, permute_153, div_40, permute_154, permute_155, permute_157, permute_158, alias_35, permute_159, permute_160, permute_165, permute_166, div_42, permute_167, permute_168, permute_169, permute_170, div_43, permute_171, permute_172, permute_174, permute_175, alias_37, permute_176, permute_177, permute_182, permute_183, div_45, permute_184, permute_185, permute_186, permute_187, div_46, permute_188, permute_189, permute_191, permute_192, alias_39, permute_193, permute_194, permute_199, permute_200, div_48, permute_201, permute_202, permute_203, permute_204, div_49, permute_205, permute_206, permute_208, permute_209, alias_41, permute_210, permute_211, permute_216, permute_217, div_51, permute_218, permute_219, permute_220, permute_221, div_52, permute_222, permute_223, permute_225, permute_226, alias_43, permute_227, permute_228, permute_233, permute_234, div_54, permute_235, permute_236, permute_237, permute_238, div_55, permute_239, permute_240, permute_242, permute_243, alias_45, permute_244, permute_245, permute_250, permute_251, div_57, permute_252, permute_253, permute_254, permute_255, div_58, permute_256, permute_257, permute_259, permute_260, alias_47, permute_261, permute_262, permute_267, permute_268, div_60, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26 = args
    args.clear()
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_101, (768, ), (1, ))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_109, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_113, (768, ), (1, ))
    assert_size_stride(primals_115, (768, ), (1, ))
    assert_size_stride(primals_117, (768, ), (1, ))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_125, (768, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_129, (768, ), (1, ))
    assert_size_stride(primals_131, (768, ), (1, ))
    assert_size_stride(primals_133, (768, ), (1, ))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_141, (768, ), (1, ))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_145, (768, ), (1, ))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_150, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_151, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_152, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_153, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_154, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_155, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_156, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_157, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_158, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_159, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_160, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_161, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(view, (1, 1024), (1024, 1))
    assert_size_stride(view_1, (1, 1024), (1024, 1))
    assert_size_stride(getitem_1, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_8, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_10, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_2, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_2, (1024, 3072), (3072, 1))
    assert_size_stride(tanh, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_14, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_8, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_21, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_23, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_10, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_6, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_1, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_27, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_16, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_34, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_36, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_18, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_10, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_2, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_40, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_24, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_47, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_49, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_26, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_14, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_3, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_53, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_32, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_60, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_62, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_34, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_18, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_4, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_66, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_40, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_73, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_75, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_42, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_22, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_5, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_79, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_48, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_86, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_88, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_50, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_26, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_6, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_92, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_56, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_99, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_101, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_58, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_30, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_7, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_105, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_64, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_112, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_114, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_66, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_34, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_8, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_118, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_72, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_125, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_127, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_74, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_38, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_9, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_131, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_80, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_138, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_140, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_82, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_42, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_10, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_144, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_88, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_151, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_153, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_90, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_46, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_11, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_157, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_96, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_219, (1024, 768), (768, 1))
    assert_size_stride(sub_37, (1, ), (1, ))
    assert_size_stride(full_default_24, (1, ), (1, ))
    assert_size_stride(permute_63, (2, 768), (768, 1))
    assert_size_stride(div_24, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_65, (768, 3072), (1, 768))
    assert_size_stride(permute_66, (3072, 1024), (1, 3072))
    assert_size_stride(permute_67, (3072, 768), (1, 3072))
    assert_size_stride(permute_68, (768, 1024), (1, 768))
    assert_size_stride(div_25, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_69, (768, 768), (1, 768))
    assert_size_stride(permute_70, (768, 1024), (1, 768))
    assert_size_stride(permute_72, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_73, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_25, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_74, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_75, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_80, (2304, 768), (1, 2304))
    assert_size_stride(permute_81, (768, 1024), (1, 768))
    assert_size_stride(div_27, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_82, (768, 3072), (1, 768))
    assert_size_stride(permute_83, (3072, 1024), (1, 3072))
    assert_size_stride(permute_84, (3072, 768), (1, 3072))
    assert_size_stride(permute_85, (768, 1024), (1, 768))
    assert_size_stride(div_28, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_86, (768, 768), (1, 768))
    assert_size_stride(permute_87, (768, 1024), (1, 768))
    assert_size_stride(permute_89, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_90, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_27, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_91, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_92, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_97, (2304, 768), (1, 2304))
    assert_size_stride(permute_98, (768, 1024), (1, 768))
    assert_size_stride(div_30, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_99, (768, 3072), (1, 768))
    assert_size_stride(permute_100, (3072, 1024), (1, 3072))
    assert_size_stride(permute_101, (3072, 768), (1, 3072))
    assert_size_stride(permute_102, (768, 1024), (1, 768))
    assert_size_stride(div_31, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_103, (768, 768), (1, 768))
    assert_size_stride(permute_104, (768, 1024), (1, 768))
    assert_size_stride(permute_106, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_107, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_29, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_108, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_109, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_114, (2304, 768), (1, 2304))
    assert_size_stride(permute_115, (768, 1024), (1, 768))
    assert_size_stride(div_33, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_116, (768, 3072), (1, 768))
    assert_size_stride(permute_117, (3072, 1024), (1, 3072))
    assert_size_stride(permute_118, (3072, 768), (1, 3072))
    assert_size_stride(permute_119, (768, 1024), (1, 768))
    assert_size_stride(div_34, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_120, (768, 768), (1, 768))
    assert_size_stride(permute_121, (768, 1024), (1, 768))
    assert_size_stride(permute_123, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_124, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_31, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_125, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_126, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_131, (2304, 768), (1, 2304))
    assert_size_stride(permute_132, (768, 1024), (1, 768))
    assert_size_stride(div_36, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_133, (768, 3072), (1, 768))
    assert_size_stride(permute_134, (3072, 1024), (1, 3072))
    assert_size_stride(permute_135, (3072, 768), (1, 3072))
    assert_size_stride(permute_136, (768, 1024), (1, 768))
    assert_size_stride(div_37, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_137, (768, 768), (1, 768))
    assert_size_stride(permute_138, (768, 1024), (1, 768))
    assert_size_stride(permute_140, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_141, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_33, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_142, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_143, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_148, (2304, 768), (1, 2304))
    assert_size_stride(permute_149, (768, 1024), (1, 768))
    assert_size_stride(div_39, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_150, (768, 3072), (1, 768))
    assert_size_stride(permute_151, (3072, 1024), (1, 3072))
    assert_size_stride(permute_152, (3072, 768), (1, 3072))
    assert_size_stride(permute_153, (768, 1024), (1, 768))
    assert_size_stride(div_40, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_154, (768, 768), (1, 768))
    assert_size_stride(permute_155, (768, 1024), (1, 768))
    assert_size_stride(permute_157, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_158, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_35, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_159, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_160, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_165, (2304, 768), (1, 2304))
    assert_size_stride(permute_166, (768, 1024), (1, 768))
    assert_size_stride(div_42, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_167, (768, 3072), (1, 768))
    assert_size_stride(permute_168, (3072, 1024), (1, 3072))
    assert_size_stride(permute_169, (3072, 768), (1, 3072))
    assert_size_stride(permute_170, (768, 1024), (1, 768))
    assert_size_stride(div_43, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_171, (768, 768), (1, 768))
    assert_size_stride(permute_172, (768, 1024), (1, 768))
    assert_size_stride(permute_174, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_175, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_37, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_176, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_177, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_182, (2304, 768), (1, 2304))
    assert_size_stride(permute_183, (768, 1024), (1, 768))
    assert_size_stride(div_45, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_184, (768, 3072), (1, 768))
    assert_size_stride(permute_185, (3072, 1024), (1, 3072))
    assert_size_stride(permute_186, (3072, 768), (1, 3072))
    assert_size_stride(permute_187, (768, 1024), (1, 768))
    assert_size_stride(div_46, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_188, (768, 768), (1, 768))
    assert_size_stride(permute_189, (768, 1024), (1, 768))
    assert_size_stride(permute_191, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_192, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_39, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_193, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_194, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_199, (2304, 768), (1, 2304))
    assert_size_stride(permute_200, (768, 1024), (1, 768))
    assert_size_stride(div_48, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_201, (768, 3072), (1, 768))
    assert_size_stride(permute_202, (3072, 1024), (1, 3072))
    assert_size_stride(permute_203, (3072, 768), (1, 3072))
    assert_size_stride(permute_204, (768, 1024), (1, 768))
    assert_size_stride(div_49, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_205, (768, 768), (1, 768))
    assert_size_stride(permute_206, (768, 1024), (1, 768))
    assert_size_stride(permute_208, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_209, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_41, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_210, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_211, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_216, (2304, 768), (1, 2304))
    assert_size_stride(permute_217, (768, 1024), (1, 768))
    assert_size_stride(div_51, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_218, (768, 3072), (1, 768))
    assert_size_stride(permute_219, (3072, 1024), (1, 3072))
    assert_size_stride(permute_220, (3072, 768), (1, 3072))
    assert_size_stride(permute_221, (768, 1024), (1, 768))
    assert_size_stride(div_52, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_222, (768, 768), (1, 768))
    assert_size_stride(permute_223, (768, 1024), (1, 768))
    assert_size_stride(permute_225, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_226, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_43, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_227, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_228, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_233, (2304, 768), (1, 2304))
    assert_size_stride(permute_234, (768, 1024), (1, 768))
    assert_size_stride(div_54, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_235, (768, 3072), (1, 768))
    assert_size_stride(permute_236, (3072, 1024), (1, 3072))
    assert_size_stride(permute_237, (3072, 768), (1, 3072))
    assert_size_stride(permute_238, (768, 1024), (1, 768))
    assert_size_stride(div_55, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_239, (768, 768), (1, 768))
    assert_size_stride(permute_240, (768, 1024), (1, 768))
    assert_size_stride(permute_242, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_243, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_45, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_244, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_245, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_250, (2304, 768), (1, 2304))
    assert_size_stride(permute_251, (768, 1024), (1, 768))
    assert_size_stride(div_57, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_252, (768, 3072), (1, 768))
    assert_size_stride(permute_253, (3072, 1024), (1, 3072))
    assert_size_stride(permute_254, (3072, 768), (1, 3072))
    assert_size_stride(permute_255, (768, 1024), (1, 768))
    assert_size_stride(div_58, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_256, (768, 768), (1, 768))
    assert_size_stride(permute_257, (768, 1024), (1, 768))
    assert_size_stride(permute_259, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_260, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_47, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_261, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_262, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_267, (2304, 768), (1, 2304))
    assert_size_stride(permute_268, (768, 1024), (1, 768))
    assert_size_stride(div_60, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(tangents_1, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(tangents_2, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_3, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_4, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_5, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_6, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_7, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_8, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_9, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_10, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_11, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_12, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_13, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_14, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_15, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_16, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_17, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_18, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_19, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_20, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_21, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_22, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_23, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_24, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_25, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_26, (1, 2), (2, 1))
    buf0 = empty((1, 1024, 2), device='cpu', dtype=torch.float32)
    cpp_fused_new_zeros_0(c_void_p(buf0.data_ptr()))
    aten.index_put_(buf0, [full_default_24, sub_37], tangents_26, True)
    del full_default_24
    del sub_37
    del tangents_26
    buf3 = empty((2, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf0, (2, 1024), (1, 2), 0), view_219, out=buf3)
    del view_219
    buf4 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf0, (1024, 2), (2, 1), 0), permute_63, out=buf4)
    del buf0
    del permute_63
    buf5 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf7 = empty((1, 1024, 768), device='cpu', dtype=torch.float32)
    buf8 = empty((768, ), device='cpu', dtype=torch.float32)
    buf9 = empty((768, ), device='cpu', dtype=torch.float32)
    buf10 = empty((1, 1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_1(c_void_p(tangents_1.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(mul_96.data_ptr()), c_void_p(div_24.data_ptr()), c_void_p(getitem_157.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()))
    del div_24
    del getitem_157
    del mul_96
    del primals_147
    del tangents_1
    buf11 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf10, (1024, 768), (768, 1), 0), permute_65, out=buf11)
    del permute_65
    buf12 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_66, reinterpret_tensor(buf10, (1024, 768), (768, 1), 0), out=buf12)
    del permute_66
    buf13 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf14 = reinterpret_tensor(buf11, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf11  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_2(c_void_p(buf14.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(addmm_46.data_ptr()), c_void_p(tanh_11.data_ptr()), c_void_p(buf13.data_ptr()))
    del addmm_46
    del tanh_11
    buf15 = reinterpret_tensor(buf10, (1024, 768), (768, 1), 0); del buf10  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf14, (1024, 3072), (3072, 1), 0), permute_67, out=buf15)
    del permute_67
    buf16 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_68, reinterpret_tensor(buf14, (1024, 3072), (3072, 1), 0), out=buf16)
    del permute_68
    buf17 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf18 = buf6; del buf6  # reuse
    buf19 = buf5; del buf5  # reuse
    buf20 = empty((768, ), device='cpu', dtype=torch.float32)
    buf21 = empty((768, ), device='cpu', dtype=torch.float32)
    buf22 = reinterpret_tensor(buf15, (1, 1024, 768), (786432, 768, 1), 0); del buf15  # reuse
    buf23 = reinterpret_tensor(buf4, (1, 1024, 768), (786432, 768, 1), 0); del buf4  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_3(c_void_p(buf22.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(mul_90.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(div_25.data_ptr()), c_void_p(getitem_153.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf23.data_ptr()))
    del div_25
    del getitem_153
    del mul_90
    del primals_145
    buf24 = reinterpret_tensor(buf7, (1024, 768), (768, 1), 0); del buf7  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf23, (1024, 768), (768, 1), 0), permute_69, out=buf24)
    del permute_69
    buf25 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_70, reinterpret_tensor(buf23, (1024, 768), (768, 1), 0), out=buf25)
    del permute_70
    buf26 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_4(c_void_p(buf23.data_ptr()), c_void_p(buf26.data_ptr()))
    buf27 = reinterpret_tensor(buf23, (12, 1024, 64), (65536, 64, 1), 0); del buf23  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_72, reinterpret_tensor(buf24, (12, 1024, 64), (64, 768, 1), 0), out=buf27)
    del permute_72
    buf28 = empty((12, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf24, (12, 1024, 64), (64, 768, 1), 0), permute_73, out=buf28)
    del permute_73
    buf29 = empty_strided((1, 12, 1024, 1), (12288, 1024, 1, 12288), device='cpu', dtype=torch.float32)
    buf30 = reinterpret_tensor(buf28, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf28  # reuse
    cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_5(c_void_p(buf30.data_ptr()), c_void_p(getitem_151.data_ptr()), c_void_p(alias_25.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(buf29.data_ptr()))
    del alias_25
    del getitem_151
    del primals_161
    buf31 = reinterpret_tensor(buf24, (12, 64, 1024), (65536, 1024, 1), 0); del buf24  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_74, reinterpret_tensor(buf30, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf31)
    del permute_74
    buf32 = empty((12, 1024, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf30, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_75, out=buf32)
    del permute_75
    buf33 = empty((1, 1024, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_6(c_void_p(buf32.data_ptr()), c_void_p(tangents_24.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(tangents_25.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf33.data_ptr()))
    del tangents_24
    del tangents_25
    buf34 = reinterpret_tensor(buf32, (1024, 768), (768, 1), 0); del buf32  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf33, (1024, 2304), (2304, 1), 0), permute_80, out=buf34)
    del permute_80
    buf35 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_81, reinterpret_tensor(buf33, (1024, 2304), (2304, 1), 0), out=buf35)
    del permute_81
    buf36 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf37 = buf19; del buf19  # reuse
    buf38 = buf18; del buf18  # reuse
    buf39 = empty((768, ), device='cpu', dtype=torch.float32)
    buf40 = empty((768, ), device='cpu', dtype=torch.float32)
    buf41 = buf22; del buf22  # reuse
    buf42 = reinterpret_tensor(buf31, (1, 1024, 768), (786432, 768, 1), 0); del buf31  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_7(c_void_p(buf41.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(mul_88.data_ptr()), c_void_p(div_27.data_ptr()), c_void_p(getitem_144.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf42.data_ptr()))
    del div_27
    del getitem_144
    del mul_88
    del primals_143
    buf43 = reinterpret_tensor(buf14, (1024, 3072), (3072, 1), 0); del buf14  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf42, (1024, 768), (768, 1), 0), permute_82, out=buf43)
    del permute_82
    buf44 = reinterpret_tensor(buf33, (3072, 768), (768, 1), 0); del buf33  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_83, reinterpret_tensor(buf42, (1024, 768), (768, 1), 0), out=buf44)
    del permute_83
    buf45 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf46 = reinterpret_tensor(buf43, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf43  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_8(c_void_p(buf46.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(addmm_42.data_ptr()), c_void_p(tanh_10.data_ptr()), c_void_p(buf45.data_ptr()))
    del addmm_42
    del tanh_10
    buf47 = reinterpret_tensor(buf42, (1024, 768), (768, 1), 0); del buf42  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf46, (1024, 3072), (3072, 1), 0), permute_84, out=buf47)
    del permute_84
    buf48 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_85, reinterpret_tensor(buf46, (1024, 3072), (3072, 1), 0), out=buf48)
    del permute_85
    buf49 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf50 = buf38; del buf38  # reuse
    buf51 = buf37; del buf37  # reuse
    buf52 = empty((768, ), device='cpu', dtype=torch.float32)
    buf53 = empty((768, ), device='cpu', dtype=torch.float32)
    buf54 = buf41; del buf41  # reuse
    buf55 = reinterpret_tensor(buf34, (1, 1024, 768), (786432, 768, 1), 0); del buf34  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_9(c_void_p(buf54.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(mul_82.data_ptr()), c_void_p(div_28.data_ptr()), c_void_p(getitem_140.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf55.data_ptr()))
    del div_28
    del getitem_140
    del mul_82
    del primals_141
    buf56 = buf47; del buf47  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf55, (1024, 768), (768, 1), 0), permute_86, out=buf56)
    del permute_86
    buf57 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_87, reinterpret_tensor(buf55, (1024, 768), (768, 1), 0), out=buf57)
    del permute_87
    buf58 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_10(c_void_p(buf55.data_ptr()), c_void_p(buf58.data_ptr()))
    buf59 = reinterpret_tensor(buf55, (12, 1024, 64), (65536, 64, 1), 0); del buf55  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_89, reinterpret_tensor(buf56, (12, 1024, 64), (64, 768, 1), 0), out=buf59)
    del permute_89
    buf60 = reinterpret_tensor(buf30, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf30  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf56, (12, 1024, 64), (64, 768, 1), 0), permute_90, out=buf60)
    del permute_90
    buf61 = buf29; del buf29  # reuse
    buf62 = reinterpret_tensor(buf60, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf60  # reuse
    cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_11(c_void_p(buf62.data_ptr()), c_void_p(getitem_138.data_ptr()), c_void_p(alias_27.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(buf61.data_ptr()))
    del alias_27
    del getitem_138
    del primals_160
    buf63 = reinterpret_tensor(buf56, (12, 64, 1024), (65536, 1024, 1), 0); del buf56  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_91, reinterpret_tensor(buf62, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf63)
    del permute_91
    buf64 = buf27; del buf27  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf62, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_92, out=buf64)
    del permute_92
    buf65 = empty((1, 1024, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_12(c_void_p(buf64.data_ptr()), c_void_p(tangents_22.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(tangents_23.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf65.data_ptr()))
    del tangents_22
    del tangents_23
    buf66 = reinterpret_tensor(buf64, (1024, 768), (768, 1), 0); del buf64  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf65, (1024, 2304), (2304, 1), 0), permute_97, out=buf66)
    del permute_97
    buf67 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_98, reinterpret_tensor(buf65, (1024, 2304), (2304, 1), 0), out=buf67)
    del permute_98
    buf68 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf69 = buf51; del buf51  # reuse
    buf70 = buf50; del buf50  # reuse
    buf71 = empty((768, ), device='cpu', dtype=torch.float32)
    buf72 = empty((768, ), device='cpu', dtype=torch.float32)
    buf73 = buf54; del buf54  # reuse
    buf74 = reinterpret_tensor(buf63, (1, 1024, 768), (786432, 768, 1), 0); del buf63  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_13(c_void_p(buf73.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(mul_80.data_ptr()), c_void_p(div_30.data_ptr()), c_void_p(getitem_131.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf74.data_ptr()))
    del div_30
    del getitem_131
    del mul_80
    del primals_139
    buf75 = reinterpret_tensor(buf46, (1024, 3072), (3072, 1), 0); del buf46  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf74, (1024, 768), (768, 1), 0), permute_99, out=buf75)
    del permute_99
    buf76 = reinterpret_tensor(buf65, (3072, 768), (768, 1), 0); del buf65  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_100, reinterpret_tensor(buf74, (1024, 768), (768, 1), 0), out=buf76)
    del permute_100
    buf77 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf78 = reinterpret_tensor(buf75, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf75  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_14(c_void_p(buf78.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(addmm_38.data_ptr()), c_void_p(tanh_9.data_ptr()), c_void_p(buf77.data_ptr()))
    del addmm_38
    del tanh_9
    buf79 = reinterpret_tensor(buf74, (1024, 768), (768, 1), 0); del buf74  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf78, (1024, 3072), (3072, 1), 0), permute_101, out=buf79)
    del permute_101
    buf80 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_102, reinterpret_tensor(buf78, (1024, 3072), (3072, 1), 0), out=buf80)
    del permute_102
    buf81 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf82 = buf70; del buf70  # reuse
    buf83 = buf69; del buf69  # reuse
    buf84 = empty((768, ), device='cpu', dtype=torch.float32)
    buf85 = empty((768, ), device='cpu', dtype=torch.float32)
    buf86 = buf73; del buf73  # reuse
    buf87 = reinterpret_tensor(buf66, (1, 1024, 768), (786432, 768, 1), 0); del buf66  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_15(c_void_p(buf86.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(mul_74.data_ptr()), c_void_p(div_31.data_ptr()), c_void_p(getitem_127.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf87.data_ptr()))
    del div_31
    del getitem_127
    del mul_74
    del primals_137
    buf88 = buf79; del buf79  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf87, (1024, 768), (768, 1), 0), permute_103, out=buf88)
    del permute_103
    buf89 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_104, reinterpret_tensor(buf87, (1024, 768), (768, 1), 0), out=buf89)
    del permute_104
    buf90 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_16(c_void_p(buf87.data_ptr()), c_void_p(buf90.data_ptr()))
    buf91 = reinterpret_tensor(buf87, (12, 1024, 64), (65536, 64, 1), 0); del buf87  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_106, reinterpret_tensor(buf88, (12, 1024, 64), (64, 768, 1), 0), out=buf91)
    del permute_106
    buf92 = reinterpret_tensor(buf62, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf62  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf88, (12, 1024, 64), (64, 768, 1), 0), permute_107, out=buf92)
    del permute_107
    buf93 = buf61; del buf61  # reuse
    buf94 = reinterpret_tensor(buf92, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf92  # reuse
    cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_17(c_void_p(buf94.data_ptr()), c_void_p(getitem_125.data_ptr()), c_void_p(alias_29.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(buf93.data_ptr()))
    del alias_29
    del getitem_125
    del primals_159
    buf95 = reinterpret_tensor(buf88, (12, 64, 1024), (65536, 1024, 1), 0); del buf88  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_108, reinterpret_tensor(buf94, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf95)
    del permute_108
    buf96 = buf59; del buf59  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf94, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_109, out=buf96)
    del permute_109
    buf97 = empty((1, 1024, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_18(c_void_p(buf96.data_ptr()), c_void_p(tangents_20.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(tangents_21.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf97.data_ptr()))
    del tangents_20
    del tangents_21
    buf98 = reinterpret_tensor(buf96, (1024, 768), (768, 1), 0); del buf96  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf97, (1024, 2304), (2304, 1), 0), permute_114, out=buf98)
    del permute_114
    buf99 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_115, reinterpret_tensor(buf97, (1024, 2304), (2304, 1), 0), out=buf99)
    del permute_115
    buf100 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf101 = buf83; del buf83  # reuse
    buf102 = buf82; del buf82  # reuse
    buf103 = empty((768, ), device='cpu', dtype=torch.float32)
    buf104 = empty((768, ), device='cpu', dtype=torch.float32)
    buf105 = buf86; del buf86  # reuse
    buf106 = reinterpret_tensor(buf95, (1, 1024, 768), (786432, 768, 1), 0); del buf95  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_19(c_void_p(buf105.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(mul_72.data_ptr()), c_void_p(div_33.data_ptr()), c_void_p(getitem_118.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf106.data_ptr()))
    del div_33
    del getitem_118
    del mul_72
    del primals_135
    buf107 = reinterpret_tensor(buf78, (1024, 3072), (3072, 1), 0); del buf78  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf106, (1024, 768), (768, 1), 0), permute_116, out=buf107)
    del permute_116
    buf108 = reinterpret_tensor(buf97, (3072, 768), (768, 1), 0); del buf97  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_117, reinterpret_tensor(buf106, (1024, 768), (768, 1), 0), out=buf108)
    del permute_117
    buf109 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf110 = reinterpret_tensor(buf107, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf107  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_20(c_void_p(buf110.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(tanh_8.data_ptr()), c_void_p(buf109.data_ptr()))
    del addmm_34
    del tanh_8
    buf111 = reinterpret_tensor(buf106, (1024, 768), (768, 1), 0); del buf106  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf110, (1024, 3072), (3072, 1), 0), permute_118, out=buf111)
    del permute_118
    buf112 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_119, reinterpret_tensor(buf110, (1024, 3072), (3072, 1), 0), out=buf112)
    del permute_119
    buf113 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf114 = buf102; del buf102  # reuse
    buf115 = buf101; del buf101  # reuse
    buf116 = empty((768, ), device='cpu', dtype=torch.float32)
    buf117 = empty((768, ), device='cpu', dtype=torch.float32)
    buf118 = buf105; del buf105  # reuse
    buf119 = reinterpret_tensor(buf98, (1, 1024, 768), (786432, 768, 1), 0); del buf98  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_21(c_void_p(buf118.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(mul_66.data_ptr()), c_void_p(div_34.data_ptr()), c_void_p(getitem_114.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf119.data_ptr()))
    del div_34
    del getitem_114
    del mul_66
    del primals_133
    buf120 = buf111; del buf111  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf119, (1024, 768), (768, 1), 0), permute_120, out=buf120)
    del permute_120
    buf121 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_121, reinterpret_tensor(buf119, (1024, 768), (768, 1), 0), out=buf121)
    del permute_121
    buf122 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_22(c_void_p(buf119.data_ptr()), c_void_p(buf122.data_ptr()))
    buf123 = reinterpret_tensor(buf119, (12, 1024, 64), (65536, 64, 1), 0); del buf119  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_123, reinterpret_tensor(buf120, (12, 1024, 64), (64, 768, 1), 0), out=buf123)
    del permute_123
    buf124 = reinterpret_tensor(buf94, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf94  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf120, (12, 1024, 64), (64, 768, 1), 0), permute_124, out=buf124)
    del permute_124
    buf125 = buf93; del buf93  # reuse
    buf126 = reinterpret_tensor(buf124, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf124  # reuse
    cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_23(c_void_p(buf126.data_ptr()), c_void_p(getitem_112.data_ptr()), c_void_p(alias_31.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(buf125.data_ptr()))
    del alias_31
    del getitem_112
    del primals_158
    buf127 = reinterpret_tensor(buf120, (12, 64, 1024), (65536, 1024, 1), 0); del buf120  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_125, reinterpret_tensor(buf126, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf127)
    del permute_125
    buf128 = buf91; del buf91  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf126, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_126, out=buf128)
    del permute_126
    buf129 = empty((1, 1024, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_24(c_void_p(buf128.data_ptr()), c_void_p(tangents_18.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(tangents_19.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf129.data_ptr()))
    del tangents_18
    del tangents_19
    buf130 = reinterpret_tensor(buf128, (1024, 768), (768, 1), 0); del buf128  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf129, (1024, 2304), (2304, 1), 0), permute_131, out=buf130)
    del permute_131
    buf131 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_132, reinterpret_tensor(buf129, (1024, 2304), (2304, 1), 0), out=buf131)
    del permute_132
    buf132 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf133 = buf115; del buf115  # reuse
    buf134 = buf114; del buf114  # reuse
    buf135 = empty((768, ), device='cpu', dtype=torch.float32)
    buf136 = empty((768, ), device='cpu', dtype=torch.float32)
    buf137 = buf118; del buf118  # reuse
    buf138 = reinterpret_tensor(buf127, (1, 1024, 768), (786432, 768, 1), 0); del buf127  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_25(c_void_p(buf137.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(mul_64.data_ptr()), c_void_p(div_36.data_ptr()), c_void_p(getitem_105.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf138.data_ptr()))
    del div_36
    del getitem_105
    del mul_64
    del primals_131
    buf139 = reinterpret_tensor(buf110, (1024, 3072), (3072, 1), 0); del buf110  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf138, (1024, 768), (768, 1), 0), permute_133, out=buf139)
    del permute_133
    buf140 = reinterpret_tensor(buf129, (3072, 768), (768, 1), 0); del buf129  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_134, reinterpret_tensor(buf138, (1024, 768), (768, 1), 0), out=buf140)
    del permute_134
    buf141 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf142 = reinterpret_tensor(buf139, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf139  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_26(c_void_p(buf142.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(addmm_30.data_ptr()), c_void_p(tanh_7.data_ptr()), c_void_p(buf141.data_ptr()))
    del addmm_30
    del tanh_7
    buf143 = reinterpret_tensor(buf138, (1024, 768), (768, 1), 0); del buf138  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf142, (1024, 3072), (3072, 1), 0), permute_135, out=buf143)
    del permute_135
    buf144 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_136, reinterpret_tensor(buf142, (1024, 3072), (3072, 1), 0), out=buf144)
    del permute_136
    buf145 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf146 = buf134; del buf134  # reuse
    buf147 = buf133; del buf133  # reuse
    buf148 = empty((768, ), device='cpu', dtype=torch.float32)
    buf149 = empty((768, ), device='cpu', dtype=torch.float32)
    buf150 = buf137; del buf137  # reuse
    buf151 = reinterpret_tensor(buf130, (1, 1024, 768), (786432, 768, 1), 0); del buf130  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_27(c_void_p(buf150.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(mul_58.data_ptr()), c_void_p(div_37.data_ptr()), c_void_p(getitem_101.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf151.data_ptr()))
    del div_37
    del getitem_101
    del mul_58
    del primals_129
    buf152 = buf143; del buf143  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf151, (1024, 768), (768, 1), 0), permute_137, out=buf152)
    del permute_137
    buf153 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_138, reinterpret_tensor(buf151, (1024, 768), (768, 1), 0), out=buf153)
    del permute_138
    buf154 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_28(c_void_p(buf151.data_ptr()), c_void_p(buf154.data_ptr()))
    buf155 = reinterpret_tensor(buf151, (12, 1024, 64), (65536, 64, 1), 0); del buf151  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_140, reinterpret_tensor(buf152, (12, 1024, 64), (64, 768, 1), 0), out=buf155)
    del permute_140
    buf156 = reinterpret_tensor(buf126, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf126  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf152, (12, 1024, 64), (64, 768, 1), 0), permute_141, out=buf156)
    del permute_141
    buf157 = buf125; del buf125  # reuse
    buf158 = reinterpret_tensor(buf156, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf156  # reuse
    cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_29(c_void_p(buf158.data_ptr()), c_void_p(getitem_99.data_ptr()), c_void_p(alias_33.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(buf157.data_ptr()))
    del alias_33
    del getitem_99
    del primals_157
    buf159 = reinterpret_tensor(buf152, (12, 64, 1024), (65536, 1024, 1), 0); del buf152  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_142, reinterpret_tensor(buf158, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf159)
    del permute_142
    buf160 = buf123; del buf123  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf158, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_143, out=buf160)
    del permute_143
    buf161 = empty((1, 1024, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_30(c_void_p(buf160.data_ptr()), c_void_p(tangents_16.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(tangents_17.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf161.data_ptr()))
    del tangents_16
    del tangents_17
    buf162 = reinterpret_tensor(buf160, (1024, 768), (768, 1), 0); del buf160  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf161, (1024, 2304), (2304, 1), 0), permute_148, out=buf162)
    del permute_148
    buf163 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_149, reinterpret_tensor(buf161, (1024, 2304), (2304, 1), 0), out=buf163)
    del permute_149
    buf164 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf165 = buf147; del buf147  # reuse
    buf166 = buf146; del buf146  # reuse
    buf167 = empty((768, ), device='cpu', dtype=torch.float32)
    buf168 = empty((768, ), device='cpu', dtype=torch.float32)
    buf169 = buf150; del buf150  # reuse
    buf170 = reinterpret_tensor(buf159, (1, 1024, 768), (786432, 768, 1), 0); del buf159  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_31(c_void_p(buf169.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(mul_56.data_ptr()), c_void_p(div_39.data_ptr()), c_void_p(getitem_92.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf170.data_ptr()))
    del div_39
    del getitem_92
    del mul_56
    del primals_127
    buf171 = reinterpret_tensor(buf142, (1024, 3072), (3072, 1), 0); del buf142  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf170, (1024, 768), (768, 1), 0), permute_150, out=buf171)
    del permute_150
    buf172 = reinterpret_tensor(buf161, (3072, 768), (768, 1), 0); del buf161  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_151, reinterpret_tensor(buf170, (1024, 768), (768, 1), 0), out=buf172)
    del permute_151
    buf173 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf174 = reinterpret_tensor(buf171, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf171  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_32(c_void_p(buf174.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(addmm_26.data_ptr()), c_void_p(tanh_6.data_ptr()), c_void_p(buf173.data_ptr()))
    del addmm_26
    del tanh_6
    buf175 = reinterpret_tensor(buf170, (1024, 768), (768, 1), 0); del buf170  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (1024, 3072), (3072, 1), 0), permute_152, out=buf175)
    del permute_152
    buf176 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_153, reinterpret_tensor(buf174, (1024, 3072), (3072, 1), 0), out=buf176)
    del permute_153
    buf177 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf178 = buf166; del buf166  # reuse
    buf179 = buf165; del buf165  # reuse
    buf180 = empty((768, ), device='cpu', dtype=torch.float32)
    buf181 = empty((768, ), device='cpu', dtype=torch.float32)
    buf182 = buf169; del buf169  # reuse
    buf183 = reinterpret_tensor(buf162, (1, 1024, 768), (786432, 768, 1), 0); del buf162  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_33(c_void_p(buf182.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(mul_50.data_ptr()), c_void_p(div_40.data_ptr()), c_void_p(getitem_88.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf183.data_ptr()))
    del div_40
    del getitem_88
    del mul_50
    del primals_125
    buf184 = buf175; del buf175  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf183, (1024, 768), (768, 1), 0), permute_154, out=buf184)
    del permute_154
    buf185 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_155, reinterpret_tensor(buf183, (1024, 768), (768, 1), 0), out=buf185)
    del permute_155
    buf186 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_34(c_void_p(buf183.data_ptr()), c_void_p(buf186.data_ptr()))
    buf187 = reinterpret_tensor(buf183, (12, 1024, 64), (65536, 64, 1), 0); del buf183  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_157, reinterpret_tensor(buf184, (12, 1024, 64), (64, 768, 1), 0), out=buf187)
    del permute_157
    buf188 = reinterpret_tensor(buf158, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf158  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf184, (12, 1024, 64), (64, 768, 1), 0), permute_158, out=buf188)
    del permute_158
    buf189 = buf157; del buf157  # reuse
    buf190 = reinterpret_tensor(buf188, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf188  # reuse
    cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_35(c_void_p(buf190.data_ptr()), c_void_p(getitem_86.data_ptr()), c_void_p(alias_35.data_ptr()), c_void_p(primals_156.data_ptr()), c_void_p(buf189.data_ptr()))
    del alias_35
    del getitem_86
    del primals_156
    buf191 = reinterpret_tensor(buf184, (12, 64, 1024), (65536, 1024, 1), 0); del buf184  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_159, reinterpret_tensor(buf190, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf191)
    del permute_159
    buf192 = buf155; del buf155  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf190, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_160, out=buf192)
    del permute_160
    buf193 = empty((1, 1024, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_36(c_void_p(buf192.data_ptr()), c_void_p(tangents_14.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(tangents_15.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf193.data_ptr()))
    del tangents_14
    del tangents_15
    buf194 = reinterpret_tensor(buf192, (1024, 768), (768, 1), 0); del buf192  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf193, (1024, 2304), (2304, 1), 0), permute_165, out=buf194)
    del permute_165
    buf195 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_166, reinterpret_tensor(buf193, (1024, 2304), (2304, 1), 0), out=buf195)
    del permute_166
    buf196 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf197 = buf179; del buf179  # reuse
    buf198 = buf178; del buf178  # reuse
    buf199 = empty((768, ), device='cpu', dtype=torch.float32)
    buf200 = empty((768, ), device='cpu', dtype=torch.float32)
    buf201 = buf182; del buf182  # reuse
    buf202 = reinterpret_tensor(buf191, (1, 1024, 768), (786432, 768, 1), 0); del buf191  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_37(c_void_p(buf201.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(mul_48.data_ptr()), c_void_p(div_42.data_ptr()), c_void_p(getitem_79.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf202.data_ptr()))
    del div_42
    del getitem_79
    del mul_48
    del primals_123
    buf203 = reinterpret_tensor(buf174, (1024, 3072), (3072, 1), 0); del buf174  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf202, (1024, 768), (768, 1), 0), permute_167, out=buf203)
    del permute_167
    buf204 = reinterpret_tensor(buf193, (3072, 768), (768, 1), 0); del buf193  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_168, reinterpret_tensor(buf202, (1024, 768), (768, 1), 0), out=buf204)
    del permute_168
    buf205 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf206 = reinterpret_tensor(buf203, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf203  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_38(c_void_p(buf206.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(tanh_5.data_ptr()), c_void_p(buf205.data_ptr()))
    del addmm_22
    del tanh_5
    buf207 = reinterpret_tensor(buf202, (1024, 768), (768, 1), 0); del buf202  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf206, (1024, 3072), (3072, 1), 0), permute_169, out=buf207)
    del permute_169
    buf208 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_170, reinterpret_tensor(buf206, (1024, 3072), (3072, 1), 0), out=buf208)
    del permute_170
    buf209 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf210 = buf198; del buf198  # reuse
    buf211 = buf197; del buf197  # reuse
    buf212 = empty((768, ), device='cpu', dtype=torch.float32)
    buf213 = empty((768, ), device='cpu', dtype=torch.float32)
    buf214 = buf201; del buf201  # reuse
    buf215 = reinterpret_tensor(buf194, (1, 1024, 768), (786432, 768, 1), 0); del buf194  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_39(c_void_p(buf214.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(mul_42.data_ptr()), c_void_p(div_43.data_ptr()), c_void_p(getitem_75.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf215.data_ptr()))
    del div_43
    del getitem_75
    del mul_42
    del primals_121
    buf216 = buf207; del buf207  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf215, (1024, 768), (768, 1), 0), permute_171, out=buf216)
    del permute_171
    buf217 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_172, reinterpret_tensor(buf215, (1024, 768), (768, 1), 0), out=buf217)
    del permute_172
    buf218 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_40(c_void_p(buf215.data_ptr()), c_void_p(buf218.data_ptr()))
    buf219 = reinterpret_tensor(buf215, (12, 1024, 64), (65536, 64, 1), 0); del buf215  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_174, reinterpret_tensor(buf216, (12, 1024, 64), (64, 768, 1), 0), out=buf219)
    del permute_174
    buf220 = reinterpret_tensor(buf190, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf190  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf216, (12, 1024, 64), (64, 768, 1), 0), permute_175, out=buf220)
    del permute_175
    buf221 = buf189; del buf189  # reuse
    buf222 = reinterpret_tensor(buf220, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf220  # reuse
    cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_41(c_void_p(buf222.data_ptr()), c_void_p(getitem_73.data_ptr()), c_void_p(alias_37.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(buf221.data_ptr()))
    del alias_37
    del getitem_73
    del primals_155
    buf223 = reinterpret_tensor(buf216, (12, 64, 1024), (65536, 1024, 1), 0); del buf216  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_176, reinterpret_tensor(buf222, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf223)
    del permute_176
    buf224 = buf187; del buf187  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf222, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_177, out=buf224)
    del permute_177
    buf225 = empty((1, 1024, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_42(c_void_p(buf224.data_ptr()), c_void_p(tangents_12.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(tangents_13.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf225.data_ptr()))
    del tangents_12
    del tangents_13
    buf226 = reinterpret_tensor(buf224, (1024, 768), (768, 1), 0); del buf224  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf225, (1024, 2304), (2304, 1), 0), permute_182, out=buf226)
    del permute_182
    buf227 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_183, reinterpret_tensor(buf225, (1024, 2304), (2304, 1), 0), out=buf227)
    del permute_183
    buf228 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf229 = buf211; del buf211  # reuse
    buf230 = buf210; del buf210  # reuse
    buf231 = empty((768, ), device='cpu', dtype=torch.float32)
    buf232 = empty((768, ), device='cpu', dtype=torch.float32)
    buf233 = buf214; del buf214  # reuse
    buf234 = reinterpret_tensor(buf223, (1, 1024, 768), (786432, 768, 1), 0); del buf223  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_43(c_void_p(buf233.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(mul_40.data_ptr()), c_void_p(div_45.data_ptr()), c_void_p(getitem_66.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf234.data_ptr()))
    del div_45
    del getitem_66
    del mul_40
    del primals_119
    buf235 = reinterpret_tensor(buf206, (1024, 3072), (3072, 1), 0); del buf206  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf234, (1024, 768), (768, 1), 0), permute_184, out=buf235)
    del permute_184
    buf236 = reinterpret_tensor(buf225, (3072, 768), (768, 1), 0); del buf225  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_185, reinterpret_tensor(buf234, (1024, 768), (768, 1), 0), out=buf236)
    del permute_185
    buf237 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf238 = reinterpret_tensor(buf235, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf235  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_44(c_void_p(buf238.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(addmm_18.data_ptr()), c_void_p(tanh_4.data_ptr()), c_void_p(buf237.data_ptr()))
    del addmm_18
    del tanh_4
    buf239 = reinterpret_tensor(buf234, (1024, 768), (768, 1), 0); del buf234  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf238, (1024, 3072), (3072, 1), 0), permute_186, out=buf239)
    del permute_186
    buf240 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_187, reinterpret_tensor(buf238, (1024, 3072), (3072, 1), 0), out=buf240)
    del permute_187
    buf241 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf242 = buf230; del buf230  # reuse
    buf243 = buf229; del buf229  # reuse
    buf244 = empty((768, ), device='cpu', dtype=torch.float32)
    buf245 = empty((768, ), device='cpu', dtype=torch.float32)
    buf246 = buf233; del buf233  # reuse
    buf247 = reinterpret_tensor(buf226, (1, 1024, 768), (786432, 768, 1), 0); del buf226  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_45(c_void_p(buf246.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(mul_34.data_ptr()), c_void_p(div_46.data_ptr()), c_void_p(getitem_62.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf247.data_ptr()))
    del div_46
    del getitem_62
    del mul_34
    del primals_117
    buf248 = buf239; del buf239  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf247, (1024, 768), (768, 1), 0), permute_188, out=buf248)
    del permute_188
    buf249 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_189, reinterpret_tensor(buf247, (1024, 768), (768, 1), 0), out=buf249)
    del permute_189
    buf250 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_46(c_void_p(buf247.data_ptr()), c_void_p(buf250.data_ptr()))
    buf251 = reinterpret_tensor(buf247, (12, 1024, 64), (65536, 64, 1), 0); del buf247  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_191, reinterpret_tensor(buf248, (12, 1024, 64), (64, 768, 1), 0), out=buf251)
    del permute_191
    buf252 = reinterpret_tensor(buf222, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf222  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf248, (12, 1024, 64), (64, 768, 1), 0), permute_192, out=buf252)
    del permute_192
    buf253 = buf221; del buf221  # reuse
    buf254 = reinterpret_tensor(buf252, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf252  # reuse
    cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_47(c_void_p(buf254.data_ptr()), c_void_p(getitem_60.data_ptr()), c_void_p(alias_39.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(buf253.data_ptr()))
    del alias_39
    del getitem_60
    del primals_154
    buf255 = reinterpret_tensor(buf248, (12, 64, 1024), (65536, 1024, 1), 0); del buf248  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_193, reinterpret_tensor(buf254, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf255)
    del permute_193
    buf256 = buf219; del buf219  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf254, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_194, out=buf256)
    del permute_194
    buf257 = empty((1, 1024, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_48(c_void_p(buf256.data_ptr()), c_void_p(tangents_10.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(tangents_11.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf257.data_ptr()))
    del tangents_10
    del tangents_11
    buf258 = reinterpret_tensor(buf256, (1024, 768), (768, 1), 0); del buf256  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf257, (1024, 2304), (2304, 1), 0), permute_199, out=buf258)
    del permute_199
    buf259 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_200, reinterpret_tensor(buf257, (1024, 2304), (2304, 1), 0), out=buf259)
    del permute_200
    buf260 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf261 = buf243; del buf243  # reuse
    buf262 = buf242; del buf242  # reuse
    buf263 = empty((768, ), device='cpu', dtype=torch.float32)
    buf264 = empty((768, ), device='cpu', dtype=torch.float32)
    buf265 = buf246; del buf246  # reuse
    buf266 = reinterpret_tensor(buf255, (1, 1024, 768), (786432, 768, 1), 0); del buf255  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_49(c_void_p(buf265.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(mul_32.data_ptr()), c_void_p(div_48.data_ptr()), c_void_p(getitem_53.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf266.data_ptr()))
    del div_48
    del getitem_53
    del mul_32
    del primals_115
    buf267 = reinterpret_tensor(buf238, (1024, 3072), (3072, 1), 0); del buf238  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf266, (1024, 768), (768, 1), 0), permute_201, out=buf267)
    del permute_201
    buf268 = reinterpret_tensor(buf257, (3072, 768), (768, 1), 0); del buf257  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_202, reinterpret_tensor(buf266, (1024, 768), (768, 1), 0), out=buf268)
    del permute_202
    buf269 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf270 = reinterpret_tensor(buf267, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf267  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_50(c_void_p(buf270.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(addmm_14.data_ptr()), c_void_p(tanh_3.data_ptr()), c_void_p(buf269.data_ptr()))
    del addmm_14
    del tanh_3
    buf271 = reinterpret_tensor(buf266, (1024, 768), (768, 1), 0); del buf266  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf270, (1024, 3072), (3072, 1), 0), permute_203, out=buf271)
    del permute_203
    buf272 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_204, reinterpret_tensor(buf270, (1024, 3072), (3072, 1), 0), out=buf272)
    del permute_204
    buf273 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf274 = buf262; del buf262  # reuse
    buf275 = buf261; del buf261  # reuse
    buf276 = empty((768, ), device='cpu', dtype=torch.float32)
    buf277 = empty((768, ), device='cpu', dtype=torch.float32)
    buf278 = buf265; del buf265  # reuse
    buf279 = reinterpret_tensor(buf258, (1, 1024, 768), (786432, 768, 1), 0); del buf258  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_51(c_void_p(buf278.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(mul_26.data_ptr()), c_void_p(div_49.data_ptr()), c_void_p(getitem_49.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf279.data_ptr()))
    del div_49
    del getitem_49
    del mul_26
    del primals_113
    buf280 = buf271; del buf271  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf279, (1024, 768), (768, 1), 0), permute_205, out=buf280)
    del permute_205
    buf281 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_206, reinterpret_tensor(buf279, (1024, 768), (768, 1), 0), out=buf281)
    del permute_206
    buf282 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_52(c_void_p(buf279.data_ptr()), c_void_p(buf282.data_ptr()))
    buf283 = reinterpret_tensor(buf279, (12, 1024, 64), (65536, 64, 1), 0); del buf279  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_208, reinterpret_tensor(buf280, (12, 1024, 64), (64, 768, 1), 0), out=buf283)
    del permute_208
    buf284 = reinterpret_tensor(buf254, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf254  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf280, (12, 1024, 64), (64, 768, 1), 0), permute_209, out=buf284)
    del permute_209
    buf285 = buf253; del buf253  # reuse
    buf286 = reinterpret_tensor(buf284, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf284  # reuse
    cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_53(c_void_p(buf286.data_ptr()), c_void_p(getitem_47.data_ptr()), c_void_p(alias_41.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(buf285.data_ptr()))
    del alias_41
    del getitem_47
    del primals_153
    buf287 = reinterpret_tensor(buf280, (12, 64, 1024), (65536, 1024, 1), 0); del buf280  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_210, reinterpret_tensor(buf286, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf287)
    del permute_210
    buf288 = buf251; del buf251  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf286, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_211, out=buf288)
    del permute_211
    buf289 = empty((1, 1024, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_54(c_void_p(buf288.data_ptr()), c_void_p(tangents_8.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(tangents_9.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf289.data_ptr()))
    del tangents_8
    del tangents_9
    buf290 = reinterpret_tensor(buf288, (1024, 768), (768, 1), 0); del buf288  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf289, (1024, 2304), (2304, 1), 0), permute_216, out=buf290)
    del permute_216
    buf291 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_217, reinterpret_tensor(buf289, (1024, 2304), (2304, 1), 0), out=buf291)
    del permute_217
    buf292 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf293 = buf275; del buf275  # reuse
    buf294 = buf274; del buf274  # reuse
    buf295 = empty((768, ), device='cpu', dtype=torch.float32)
    buf296 = empty((768, ), device='cpu', dtype=torch.float32)
    buf297 = buf278; del buf278  # reuse
    buf298 = reinterpret_tensor(buf287, (1, 1024, 768), (786432, 768, 1), 0); del buf287  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_55(c_void_p(buf297.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(mul_24.data_ptr()), c_void_p(div_51.data_ptr()), c_void_p(getitem_40.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf298.data_ptr()))
    del div_51
    del getitem_40
    del mul_24
    del primals_111
    buf299 = reinterpret_tensor(buf270, (1024, 3072), (3072, 1), 0); del buf270  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf298, (1024, 768), (768, 1), 0), permute_218, out=buf299)
    del permute_218
    buf300 = reinterpret_tensor(buf289, (3072, 768), (768, 1), 0); del buf289  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_219, reinterpret_tensor(buf298, (1024, 768), (768, 1), 0), out=buf300)
    del permute_219
    buf301 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf302 = reinterpret_tensor(buf299, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf299  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_56(c_void_p(buf302.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(tanh_2.data_ptr()), c_void_p(buf301.data_ptr()))
    del addmm_10
    del tanh_2
    buf303 = reinterpret_tensor(buf298, (1024, 768), (768, 1), 0); del buf298  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf302, (1024, 3072), (3072, 1), 0), permute_220, out=buf303)
    del permute_220
    buf304 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_221, reinterpret_tensor(buf302, (1024, 3072), (3072, 1), 0), out=buf304)
    del permute_221
    buf305 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf306 = buf294; del buf294  # reuse
    buf307 = buf293; del buf293  # reuse
    buf308 = empty((768, ), device='cpu', dtype=torch.float32)
    buf309 = empty((768, ), device='cpu', dtype=torch.float32)
    buf310 = buf297; del buf297  # reuse
    buf311 = reinterpret_tensor(buf290, (1, 1024, 768), (786432, 768, 1), 0); del buf290  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_57(c_void_p(buf310.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(mul_18.data_ptr()), c_void_p(div_52.data_ptr()), c_void_p(getitem_36.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf311.data_ptr()))
    del div_52
    del getitem_36
    del mul_18
    del primals_109
    buf312 = buf303; del buf303  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf311, (1024, 768), (768, 1), 0), permute_222, out=buf312)
    del permute_222
    buf313 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_223, reinterpret_tensor(buf311, (1024, 768), (768, 1), 0), out=buf313)
    del permute_223
    buf314 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_58(c_void_p(buf311.data_ptr()), c_void_p(buf314.data_ptr()))
    buf315 = reinterpret_tensor(buf311, (12, 1024, 64), (65536, 64, 1), 0); del buf311  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_225, reinterpret_tensor(buf312, (12, 1024, 64), (64, 768, 1), 0), out=buf315)
    del permute_225
    buf316 = reinterpret_tensor(buf286, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf286  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf312, (12, 1024, 64), (64, 768, 1), 0), permute_226, out=buf316)
    del permute_226
    buf317 = buf285; del buf285  # reuse
    buf318 = reinterpret_tensor(buf316, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf316  # reuse
    cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_59(c_void_p(buf318.data_ptr()), c_void_p(getitem_34.data_ptr()), c_void_p(alias_43.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(buf317.data_ptr()))
    del alias_43
    del getitem_34
    del primals_152
    buf319 = reinterpret_tensor(buf312, (12, 64, 1024), (65536, 1024, 1), 0); del buf312  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_227, reinterpret_tensor(buf318, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf319)
    del permute_227
    buf320 = buf283; del buf283  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf318, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_228, out=buf320)
    del permute_228
    buf321 = empty((1, 1024, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_60(c_void_p(buf320.data_ptr()), c_void_p(tangents_6.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(tangents_7.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf321.data_ptr()))
    del tangents_6
    del tangents_7
    buf322 = reinterpret_tensor(buf320, (1024, 768), (768, 1), 0); del buf320  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf321, (1024, 2304), (2304, 1), 0), permute_233, out=buf322)
    del permute_233
    buf323 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_234, reinterpret_tensor(buf321, (1024, 2304), (2304, 1), 0), out=buf323)
    del permute_234
    buf324 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf325 = buf307; del buf307  # reuse
    buf326 = buf306; del buf306  # reuse
    buf327 = empty((768, ), device='cpu', dtype=torch.float32)
    buf328 = empty((768, ), device='cpu', dtype=torch.float32)
    buf329 = buf310; del buf310  # reuse
    buf330 = reinterpret_tensor(buf319, (1, 1024, 768), (786432, 768, 1), 0); del buf319  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_61(c_void_p(buf329.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(mul_16.data_ptr()), c_void_p(div_54.data_ptr()), c_void_p(getitem_27.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf330.data_ptr()))
    del div_54
    del getitem_27
    del mul_16
    del primals_107
    buf331 = reinterpret_tensor(buf302, (1024, 3072), (3072, 1), 0); del buf302  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf330, (1024, 768), (768, 1), 0), permute_235, out=buf331)
    del permute_235
    buf332 = reinterpret_tensor(buf321, (3072, 768), (768, 1), 0); del buf321  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_236, reinterpret_tensor(buf330, (1024, 768), (768, 1), 0), out=buf332)
    del permute_236
    buf333 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf334 = reinterpret_tensor(buf331, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf331  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_62(c_void_p(buf334.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(addmm_6.data_ptr()), c_void_p(tanh_1.data_ptr()), c_void_p(buf333.data_ptr()))
    del addmm_6
    del tanh_1
    buf335 = reinterpret_tensor(buf330, (1024, 768), (768, 1), 0); del buf330  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf334, (1024, 3072), (3072, 1), 0), permute_237, out=buf335)
    del permute_237
    buf336 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_238, reinterpret_tensor(buf334, (1024, 3072), (3072, 1), 0), out=buf336)
    del permute_238
    buf337 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf338 = buf326; del buf326  # reuse
    buf339 = buf325; del buf325  # reuse
    buf340 = empty((768, ), device='cpu', dtype=torch.float32)
    buf341 = empty((768, ), device='cpu', dtype=torch.float32)
    buf342 = buf329; del buf329  # reuse
    buf343 = reinterpret_tensor(buf322, (1, 1024, 768), (786432, 768, 1), 0); del buf322  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_63(c_void_p(buf342.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(mul_10.data_ptr()), c_void_p(div_55.data_ptr()), c_void_p(getitem_23.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf343.data_ptr()))
    del div_55
    del getitem_23
    del mul_10
    del primals_105
    buf344 = buf335; del buf335  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf343, (1024, 768), (768, 1), 0), permute_239, out=buf344)
    del permute_239
    buf345 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_240, reinterpret_tensor(buf343, (1024, 768), (768, 1), 0), out=buf345)
    del permute_240
    buf346 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_64(c_void_p(buf343.data_ptr()), c_void_p(buf346.data_ptr()))
    buf347 = reinterpret_tensor(buf343, (12, 1024, 64), (65536, 64, 1), 0); del buf343  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_242, reinterpret_tensor(buf344, (12, 1024, 64), (64, 768, 1), 0), out=buf347)
    del permute_242
    buf348 = reinterpret_tensor(buf318, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf318  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf344, (12, 1024, 64), (64, 768, 1), 0), permute_243, out=buf348)
    del permute_243
    buf349 = buf317; del buf317  # reuse
    buf350 = reinterpret_tensor(buf348, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf348  # reuse
    cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_65(c_void_p(buf350.data_ptr()), c_void_p(getitem_21.data_ptr()), c_void_p(alias_45.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(buf349.data_ptr()))
    del alias_45
    del getitem_21
    del primals_151
    buf351 = reinterpret_tensor(buf344, (12, 64, 1024), (65536, 1024, 1), 0); del buf344  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_244, reinterpret_tensor(buf350, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf351)
    del permute_244
    buf352 = buf315; del buf315  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf350, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_245, out=buf352)
    del permute_245
    buf353 = empty((1, 1024, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_66(c_void_p(buf352.data_ptr()), c_void_p(tangents_4.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(tangents_5.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf353.data_ptr()))
    del tangents_4
    del tangents_5
    buf354 = reinterpret_tensor(buf352, (1024, 768), (768, 1), 0); del buf352  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf353, (1024, 2304), (2304, 1), 0), permute_250, out=buf354)
    del permute_250
    buf355 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_251, reinterpret_tensor(buf353, (1024, 2304), (2304, 1), 0), out=buf355)
    del permute_251
    buf356 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf357 = buf339; del buf339  # reuse
    buf358 = buf338; del buf338  # reuse
    buf359 = empty((768, ), device='cpu', dtype=torch.float32)
    buf360 = empty((768, ), device='cpu', dtype=torch.float32)
    buf361 = buf342; del buf342  # reuse
    buf362 = reinterpret_tensor(buf351, (1, 1024, 768), (786432, 768, 1), 0); del buf351  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_67(c_void_p(buf361.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(mul_8.data_ptr()), c_void_p(div_57.data_ptr()), c_void_p(getitem_14.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf362.data_ptr()))
    del div_57
    del getitem_14
    del mul_8
    del primals_103
    buf363 = reinterpret_tensor(buf334, (1024, 3072), (3072, 1), 0); del buf334  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf362, (1024, 768), (768, 1), 0), permute_252, out=buf363)
    del permute_252
    buf364 = reinterpret_tensor(buf353, (3072, 768), (768, 1), 0); del buf353  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_253, reinterpret_tensor(buf362, (1024, 768), (768, 1), 0), out=buf364)
    del permute_253
    buf365 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf366 = reinterpret_tensor(buf363, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf363  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_68(c_void_p(buf366.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(addmm_2.data_ptr()), c_void_p(tanh.data_ptr()), c_void_p(buf365.data_ptr()))
    del addmm_2
    del tanh
    buf367 = reinterpret_tensor(buf362, (1024, 768), (768, 1), 0); del buf362  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf366, (1024, 3072), (3072, 1), 0), permute_254, out=buf367)
    del permute_254
    buf368 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_255, reinterpret_tensor(buf366, (1024, 3072), (3072, 1), 0), out=buf368)
    del permute_255
    buf369 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf370 = buf358; del buf358  # reuse
    buf371 = buf357; del buf357  # reuse
    buf372 = empty((768, ), device='cpu', dtype=torch.float32)
    buf373 = empty((768, ), device='cpu', dtype=torch.float32)
    buf374 = buf361; del buf361  # reuse
    buf375 = reinterpret_tensor(buf354, (1, 1024, 768), (786432, 768, 1), 0); del buf354  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_69(c_void_p(buf374.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(mul_2.data_ptr()), c_void_p(div_58.data_ptr()), c_void_p(getitem_10.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf375.data_ptr()))
    del buf366
    del div_58
    del getitem_10
    del mul_2
    del primals_101
    buf376 = buf367; del buf367  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf375, (1024, 768), (768, 1), 0), permute_256, out=buf376)
    del permute_256
    buf377 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_257, reinterpret_tensor(buf375, (1024, 768), (768, 1), 0), out=buf377)
    del permute_257
    buf378 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_70(c_void_p(buf375.data_ptr()), c_void_p(buf378.data_ptr()))
    buf379 = reinterpret_tensor(buf375, (12, 1024, 64), (65536, 64, 1), 0); del buf375  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_259, reinterpret_tensor(buf376, (12, 1024, 64), (64, 768, 1), 0), out=buf379)
    del permute_259
    buf380 = reinterpret_tensor(buf350, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf350  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf376, (12, 1024, 64), (64, 768, 1), 0), permute_260, out=buf380)
    del permute_260
    buf381 = buf349; del buf349  # reuse
    buf382 = reinterpret_tensor(buf380, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf380  # reuse
    cpp_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_71(c_void_p(buf382.data_ptr()), c_void_p(getitem_8.data_ptr()), c_void_p(alias_47.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(buf381.data_ptr()))
    del alias_47
    del buf381
    del getitem_8
    del primals_150
    buf383 = reinterpret_tensor(buf376, (12, 64, 1024), (65536, 1024, 1), 0); del buf376  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_261, reinterpret_tensor(buf382, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf383)
    del permute_261
    buf384 = buf347; del buf347  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf382, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_262, out=buf384)
    del buf382
    del permute_262
    buf385 = empty((1, 1024, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_72(c_void_p(buf384.data_ptr()), c_void_p(tangents_2.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(tangents_3.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf385.data_ptr()))
    del tangents_2
    del tangents_3
    buf386 = reinterpret_tensor(buf384, (1024, 768), (768, 1), 0); del buf384  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf385, (1024, 2304), (2304, 1), 0), permute_267, out=buf386)
    del permute_267
    buf387 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_268, reinterpret_tensor(buf385, (1024, 2304), (2304, 1), 0), out=buf387)
    del permute_268
    buf388 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf389 = buf371; del buf371  # reuse
    buf390 = buf370; del buf370  # reuse
    buf391 = empty((768, ), device='cpu', dtype=torch.float32)
    buf392 = empty((768, ), device='cpu', dtype=torch.float32)
    buf393 = buf374; del buf374  # reuse
    buf399 = reinterpret_tensor(buf383, (1, 1024, 768), (786432, 768, 1), 0); del buf383  # reuse
    buf394 = reinterpret_tensor(buf379, (1024, 768), (768, 1), 0); del buf379  # reuse
    buf395 = buf393; del buf393  # reuse
    cpp_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_scalar_tensor_sum_73(c_void_p(buf395.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(mul.data_ptr()), c_void_p(div_60.data_ptr()), c_void_p(getitem_1.data_ptr()), c_void_p(view.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf394.data_ptr()))
    del buf385
    del buf386
    del buf389
    del buf390
    del div_60
    del getitem_1
    del mul
    del primals_99
    aten.index_put_(buf394, [view_1], buf395, True)
    del buf395
    del view_1
    buf398 = empty((50257, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_74(c_void_p(buf398.data_ptr()))
    aten.index_put_(buf398, [view], buf399, True)
    del buf399
    del view
    return (reinterpret_tensor(buf388, (2304, ), (1, ), 0), buf387, reinterpret_tensor(buf378, (768, ), (1, ), 0), buf377, reinterpret_tensor(buf369, (3072, ), (1, ), 0), buf368, reinterpret_tensor(buf365, (768, ), (1, ), 0), buf364, reinterpret_tensor(buf356, (2304, ), (1, ), 0), buf355, reinterpret_tensor(buf346, (768, ), (1, ), 0), buf345, reinterpret_tensor(buf337, (3072, ), (1, ), 0), buf336, reinterpret_tensor(buf333, (768, ), (1, ), 0), buf332, reinterpret_tensor(buf324, (2304, ), (1, ), 0), buf323, reinterpret_tensor(buf314, (768, ), (1, ), 0), buf313, reinterpret_tensor(buf305, (3072, ), (1, ), 0), buf304, reinterpret_tensor(buf301, (768, ), (1, ), 0), buf300, reinterpret_tensor(buf292, (2304, ), (1, ), 0), buf291, reinterpret_tensor(buf282, (768, ), (1, ), 0), buf281, reinterpret_tensor(buf273, (3072, ), (1, ), 0), buf272, reinterpret_tensor(buf269, (768, ), (1, ), 0), buf268, reinterpret_tensor(buf260, (2304, ), (1, ), 0), buf259, reinterpret_tensor(buf250, (768, ), (1, ), 0), buf249, reinterpret_tensor(buf241, (3072, ), (1, ), 0), buf240, reinterpret_tensor(buf237, (768, ), (1, ), 0), buf236, reinterpret_tensor(buf228, (2304, ), (1, ), 0), buf227, reinterpret_tensor(buf218, (768, ), (1, ), 0), buf217, reinterpret_tensor(buf209, (3072, ), (1, ), 0), buf208, reinterpret_tensor(buf205, (768, ), (1, ), 0), buf204, reinterpret_tensor(buf196, (2304, ), (1, ), 0), buf195, reinterpret_tensor(buf186, (768, ), (1, ), 0), buf185, reinterpret_tensor(buf177, (3072, ), (1, ), 0), buf176, reinterpret_tensor(buf173, (768, ), (1, ), 0), buf172, reinterpret_tensor(buf164, (2304, ), (1, ), 0), buf163, reinterpret_tensor(buf154, (768, ), (1, ), 0), buf153, reinterpret_tensor(buf145, (3072, ), (1, ), 0), buf144, reinterpret_tensor(buf141, (768, ), (1, ), 0), buf140, reinterpret_tensor(buf132, (2304, ), (1, ), 0), buf131, reinterpret_tensor(buf122, (768, ), (1, ), 0), buf121, reinterpret_tensor(buf113, (3072, ), (1, ), 0), buf112, reinterpret_tensor(buf109, (768, ), (1, ), 0), buf108, reinterpret_tensor(buf100, (2304, ), (1, ), 0), buf99, reinterpret_tensor(buf90, (768, ), (1, ), 0), buf89, reinterpret_tensor(buf81, (3072, ), (1, ), 0), buf80, reinterpret_tensor(buf77, (768, ), (1, ), 0), buf76, reinterpret_tensor(buf68, (2304, ), (1, ), 0), buf67, reinterpret_tensor(buf58, (768, ), (1, ), 0), buf57, reinterpret_tensor(buf49, (3072, ), (1, ), 0), buf48, reinterpret_tensor(buf45, (768, ), (1, ), 0), buf44, reinterpret_tensor(buf36, (2304, ), (1, ), 0), buf35, reinterpret_tensor(buf26, (768, ), (1, ), 0), buf25, reinterpret_tensor(buf17, (3072, ), (1, ), 0), buf16, reinterpret_tensor(buf13, (768, ), (1, ), 0), buf12, buf398, buf394, buf391, buf392, buf372, buf373, buf359, buf360, buf340, buf341, buf327, buf328, buf308, buf309, buf295, buf296, buf276, buf277, buf263, buf264, buf244, buf245, buf231, buf232, buf212, buf213, buf199, buf200, buf180, buf181, buf167, buf168, buf148, buf149, buf135, buf136, buf116, buf117, buf103, buf104, buf84, buf85, buf71, buf72, buf52, buf53, buf39, buf40, buf20, buf21, buf8, buf9, reinterpret_tensor(buf3, (2, 768), (768, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_99 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    primals_151 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    primals_152 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    primals_153 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    primals_154 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    primals_155 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    primals_156 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    primals_157 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    primals_158 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    primals_159 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    primals_160 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    primals_161 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    view = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    view_1 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    getitem_1 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    getitem_8 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    getitem_10 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_2 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    addmm_2 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_14 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_8 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    getitem_21 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    getitem_23 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_10 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    addmm_6 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_1 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_27 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_16 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    getitem_34 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    getitem_36 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_18 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_2 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_40 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_24 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    getitem_47 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    getitem_49 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_26 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    addmm_14 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_3 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_53 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_32 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    getitem_60 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    getitem_62 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_34 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    addmm_18 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_4 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_66 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_40 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    getitem_73 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    getitem_75 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_42 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_5 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_79 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_48 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    getitem_86 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    getitem_88 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_50 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    addmm_26 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_6 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_92 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_56 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    getitem_99 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    getitem_101 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_58 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    addmm_30 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_7 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_105 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_64 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    getitem_112 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    getitem_114 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_66 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_8 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_118 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_72 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    getitem_125 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    getitem_127 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_74 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    addmm_38 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_9 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_131 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_80 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    getitem_138 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    getitem_140 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_82 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    addmm_42 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_10 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_144 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_88 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    getitem_151 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    getitem_153 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_90 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    addmm_46 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_11 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cpu', dtype=torch.float32)
    getitem_157 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.bool)
    mul_96 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    view_219 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    sub_37 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    full_default_24 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    permute_63 = rand_strided((2, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_65 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_66 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_67 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_68 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_25 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_69 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_70 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_72 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_73 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    alias_25 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_74 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    permute_75 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cpu', dtype=torch.float32)
    permute_80 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_81 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_27 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_82 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_83 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_84 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_85 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_28 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_86 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_87 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_89 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_90 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    alias_27 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_91 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    permute_92 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cpu', dtype=torch.float32)
    permute_97 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_98 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_30 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_99 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_100 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_101 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_102 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_31 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_103 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_104 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_106 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_107 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    alias_29 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_108 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    permute_109 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cpu', dtype=torch.float32)
    permute_114 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_115 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_33 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_116 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_117 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_118 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_119 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_34 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_120 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_121 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_123 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_124 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    alias_31 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_125 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    permute_126 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cpu', dtype=torch.float32)
    permute_131 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_132 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_36 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_133 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_134 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_135 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_136 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_37 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_137 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_138 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_140 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_141 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    alias_33 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_142 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    permute_143 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cpu', dtype=torch.float32)
    permute_148 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_149 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_39 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_150 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_151 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_152 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_153 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_40 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_154 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_155 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_157 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_158 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    alias_35 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_159 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    permute_160 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cpu', dtype=torch.float32)
    permute_165 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_166 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_42 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_167 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_168 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_169 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_170 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_43 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_171 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_172 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_174 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_175 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    alias_37 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_176 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    permute_177 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cpu', dtype=torch.float32)
    permute_182 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_183 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_45 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_184 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_185 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_186 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_187 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_46 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_188 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_189 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_191 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_192 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    alias_39 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_193 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    permute_194 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cpu', dtype=torch.float32)
    permute_199 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_200 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_48 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_201 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_202 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_203 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_204 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_49 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_205 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_206 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_208 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_209 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    alias_41 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_210 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    permute_211 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cpu', dtype=torch.float32)
    permute_216 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_217 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_51 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_218 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_219 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_220 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_221 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_52 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_222 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_223 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_225 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_226 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    alias_43 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_227 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    permute_228 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cpu', dtype=torch.float32)
    permute_233 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_234 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_54 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_235 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_236 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_237 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_238 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_55 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_239 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_240 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_242 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_243 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    alias_45 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_244 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    permute_245 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cpu', dtype=torch.float32)
    permute_250 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_251 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_57 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_252 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_253 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_254 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_255 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_58 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    permute_256 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_257 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_259 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cpu', dtype=torch.float32)
    permute_260 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    alias_47 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    permute_261 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cpu', dtype=torch.float32)
    permute_262 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cpu', dtype=torch.float32)
    permute_267 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_268 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_60 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_3 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_4 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_5 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_6 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_7 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_8 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_9 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_10 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_11 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_12 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_13 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_14 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_15 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_16 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_17 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_18 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_19 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_20 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_21 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_22 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_23 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_24 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_25 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cpu', dtype=torch.float32)
    tangents_26 = rand_strided((1, 2), (2, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, view, view_1, getitem_1, mul, getitem_8, getitem_10, mul_2, addmm_2, tanh, getitem_14, mul_8, getitem_21, getitem_23, mul_10, addmm_6, tanh_1, getitem_27, mul_16, getitem_34, getitem_36, mul_18, addmm_10, tanh_2, getitem_40, mul_24, getitem_47, getitem_49, mul_26, addmm_14, tanh_3, getitem_53, mul_32, getitem_60, getitem_62, mul_34, addmm_18, tanh_4, getitem_66, mul_40, getitem_73, getitem_75, mul_42, addmm_22, tanh_5, getitem_79, mul_48, getitem_86, getitem_88, mul_50, addmm_26, tanh_6, getitem_92, mul_56, getitem_99, getitem_101, mul_58, addmm_30, tanh_7, getitem_105, mul_64, getitem_112, getitem_114, mul_66, addmm_34, tanh_8, getitem_118, mul_72, getitem_125, getitem_127, mul_74, addmm_38, tanh_9, getitem_131, mul_80, getitem_138, getitem_140, mul_82, addmm_42, tanh_10, getitem_144, mul_88, getitem_151, getitem_153, mul_90, addmm_46, tanh_11, getitem_157, mul_96, view_219, sub_37, full_default_24, permute_63, div_24, permute_65, permute_66, permute_67, permute_68, div_25, permute_69, permute_70, permute_72, permute_73, alias_25, permute_74, permute_75, permute_80, permute_81, div_27, permute_82, permute_83, permute_84, permute_85, div_28, permute_86, permute_87, permute_89, permute_90, alias_27, permute_91, permute_92, permute_97, permute_98, div_30, permute_99, permute_100, permute_101, permute_102, div_31, permute_103, permute_104, permute_106, permute_107, alias_29, permute_108, permute_109, permute_114, permute_115, div_33, permute_116, permute_117, permute_118, permute_119, div_34, permute_120, permute_121, permute_123, permute_124, alias_31, permute_125, permute_126, permute_131, permute_132, div_36, permute_133, permute_134, permute_135, permute_136, div_37, permute_137, permute_138, permute_140, permute_141, alias_33, permute_142, permute_143, permute_148, permute_149, div_39, permute_150, permute_151, permute_152, permute_153, div_40, permute_154, permute_155, permute_157, permute_158, alias_35, permute_159, permute_160, permute_165, permute_166, div_42, permute_167, permute_168, permute_169, permute_170, div_43, permute_171, permute_172, permute_174, permute_175, alias_37, permute_176, permute_177, permute_182, permute_183, div_45, permute_184, permute_185, permute_186, permute_187, div_46, permute_188, permute_189, permute_191, permute_192, alias_39, permute_193, permute_194, permute_199, permute_200, div_48, permute_201, permute_202, permute_203, permute_204, div_49, permute_205, permute_206, permute_208, permute_209, alias_41, permute_210, permute_211, permute_216, permute_217, div_51, permute_218, permute_219, permute_220, permute_221, div_52, permute_222, permute_223, permute_225, permute_226, alias_43, permute_227, permute_228, permute_233, permute_234, div_54, permute_235, permute_236, permute_237, permute_238, div_55, permute_239, permute_240, permute_242, permute_243, alias_45, permute_244, permute_245, permute_250, permute_251, div_57, permute_252, permute_253, permute_254, permute_255, div_58, permute_256, permute_257, permute_259, permute_260, alias_47, permute_261, permute_262, permute_267, permute_268, div_60, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('GPT2ForSequenceClassification', benchmark_compiled_module)
