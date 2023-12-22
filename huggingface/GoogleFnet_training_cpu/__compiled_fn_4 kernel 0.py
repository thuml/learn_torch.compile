
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


cpp_fused_nll_loss_backward_nll_loss_forward_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       float* out_ptr0,
                       long* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384000L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<long>(-100);
                    auto tmp2 = tmp0 != tmp1;
                    auto tmp3 = static_cast<long>(0);
                    auto tmp4 = tmp2 ? tmp0 : tmp3;
                    out_ptr1[static_cast<long>(x0)] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32000L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32000L*x0)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0)];
                        auto tmp4 = in_ptr2[static_cast<long>(0L)];
                        auto tmp5 = in_ptr3[static_cast<long>(0L)];
                        auto tmp2 = static_cast<int>(-100);
                        auto tmp3 = tmp1 != tmp2;
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(0.0);
                        auto tmp8 = tmp3 ? tmp6 : tmp7;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp0 * tmp9;
                        tmp_acc0_vec = tmp_acc0_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32000L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (32000L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32000L*x0)));
                    auto tmp2 = in_ptr1[static_cast<long>(x0)];
                    auto tmp5 = in_ptr2[static_cast<long>(0L)];
                    auto tmp6 = in_ptr3[static_cast<long>(0L)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (32000L*x0)));
                    auto tmp14 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = static_cast<int>(-100);
                    auto tmp4 = tmp2 != tmp3;
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp8 = static_cast<float>(0.0);
                    auto tmp9 = tmp4 ? tmp7 : tmp8;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp1 * tmp10;
                    auto tmp13 = tmp12.exp();
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp11 - tmp16;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (32000L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_sum_tanh_backward_2 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr4,
                       float* out_ptr5)
{
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32000L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32000L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp12 = in_ptr5[static_cast<long>(x0)];
                        auto tmp15 = in_ptr6[static_cast<long>(x0)];
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = static_cast<float>(0.5);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = static_cast<float>(1.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 + tmp9;
                        auto tmp11 = tmp6 * tmp10;
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 - tmp13;
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp18 = tmp2 * tmp17;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp18;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = in_ptr5[static_cast<long>(x0)];
                    auto tmp25 = out_ptr2[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(tmp1);
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 - tmp9;
                    auto tmp12 = static_cast<float>(0.5);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp16 = static_cast<float>(1.0);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 + tmp17;
                    auto tmp19 = tmp14 * tmp18;
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = tmp19 - tmp21;
                    auto tmp23 = at::vec::Vectorized<float>(tmp0);
                    auto tmp24 = tmp22 * tmp23;
                    auto tmp26 = at::vec::Vectorized<float>(tmp25);
                    auto tmp27 = tmp24 * tmp26;
                    auto tmp28 = tmp10 - tmp27;
                    auto tmp29 = at::vec::Vectorized<float>(tmp2);
                    auto tmp30 = tmp29 * tmp28;
                    tmp30.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp10 = in_ptr5[static_cast<long>(x1)];
                        auto tmp13 = in_ptr6[static_cast<long>(x1)];
                        auto tmp2 = static_cast<float>(0.5);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 + tmp7;
                        auto tmp9 = tmp4 * tmp8;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 - tmp11;
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp12 * tmp14;
                        auto tmp16 = tmp0 * tmp15;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
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


cpp_fused_native_dropout_backward_native_layer_norm_backward_sum_3 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp11 = out_ptr2[static_cast<long>(x0)];
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
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = in_ptr5[static_cast<long>(x0)];
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


cpp_fused_add_mul_pow_sum_tanh_backward_4 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    auto tmp3 = out_ptr3[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<int>(x1);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp2 ? tmp3 : tmp4;
                    out_ptr6[static_cast<long>(x1 + (2L*x0))] = tmp5;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp9 = tmp6 * tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0 + x0_inner)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp9 = out_ptr0[static_cast<long>(x0 + x0_inner)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp13 = out_ptr1[static_cast<long>(x0 + x0_inner)];
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
                        tmp18.store(out_ptr2 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x0) + (2L*x0_inner) + (1536L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_add_mul_pow_sum_tanh_backward_7 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    auto tmp3 = out_ptr3[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<int>(x1);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp2 ? tmp3 : tmp4;
                    out_ptr6[static_cast<long>(x1 + (2L*x0))] = tmp5;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp9 = tmp6 * tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0 + x0_inner)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp9 = out_ptr0[static_cast<long>(x0 + x0_inner)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp13 = out_ptr1[static_cast<long>(x0 + x0_inner)];
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
                        tmp18.store(out_ptr2 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x0) + (2L*x0_inner) + (1536L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_add_mul_pow_sum_tanh_backward_10 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    auto tmp3 = out_ptr3[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<int>(x1);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp2 ? tmp3 : tmp4;
                    out_ptr6[static_cast<long>(x1 + (2L*x0))] = tmp5;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp9 = tmp6 * tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0 + x0_inner)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp9 = out_ptr0[static_cast<long>(x0 + x0_inner)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp13 = out_ptr1[static_cast<long>(x0 + x0_inner)];
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
                        tmp18.store(out_ptr2 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x0) + (2L*x0_inner) + (1536L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_add_mul_pow_sum_tanh_backward_13 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    auto tmp3 = out_ptr3[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<int>(x1);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp2 ? tmp3 : tmp4;
                    out_ptr6[static_cast<long>(x1 + (2L*x0))] = tmp5;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp9 = tmp6 * tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0 + x0_inner)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp9 = out_ptr0[static_cast<long>(x0 + x0_inner)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp13 = out_ptr1[static_cast<long>(x0 + x0_inner)];
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
                        tmp18.store(out_ptr2 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x0) + (2L*x0_inner) + (1536L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_add_mul_pow_sum_tanh_backward_16 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    auto tmp3 = out_ptr3[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<int>(x1);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp2 ? tmp3 : tmp4;
                    out_ptr6[static_cast<long>(x1 + (2L*x0))] = tmp5;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp9 = tmp6 * tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0 + x0_inner)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp9 = out_ptr0[static_cast<long>(x0 + x0_inner)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp13 = out_ptr1[static_cast<long>(x0 + x0_inner)];
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
                        tmp18.store(out_ptr2 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x0) + (2L*x0_inner) + (1536L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_add_mul_pow_sum_tanh_backward_19 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    auto tmp3 = out_ptr3[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<int>(x1);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp2 ? tmp3 : tmp4;
                    out_ptr6[static_cast<long>(x1 + (2L*x0))] = tmp5;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp9 = tmp6 * tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0 + x0_inner)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp9 = out_ptr0[static_cast<long>(x0 + x0_inner)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp13 = out_ptr1[static_cast<long>(x0 + x0_inner)];
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
                        tmp18.store(out_ptr2 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x0) + (2L*x0_inner) + (1536L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_add_mul_pow_sum_tanh_backward_22 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    auto tmp3 = out_ptr3[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<int>(x1);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp2 ? tmp3 : tmp4;
                    out_ptr6[static_cast<long>(x1 + (2L*x0))] = tmp5;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp9 = tmp6 * tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0 + x0_inner)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp9 = out_ptr0[static_cast<long>(x0 + x0_inner)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp13 = out_ptr1[static_cast<long>(x0 + x0_inner)];
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
                        tmp18.store(out_ptr2 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x0) + (2L*x0_inner) + (1536L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_add_mul_pow_sum_tanh_backward_25 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    auto tmp3 = out_ptr3[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<int>(x1);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp2 ? tmp3 : tmp4;
                    out_ptr6[static_cast<long>(x1 + (2L*x0))] = tmp5;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp9 = tmp6 * tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0 + x0_inner)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp9 = out_ptr0[static_cast<long>(x0 + x0_inner)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp13 = out_ptr1[static_cast<long>(x0 + x0_inner)];
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
                        tmp18.store(out_ptr2 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x0) + (2L*x0_inner) + (1536L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_add_mul_pow_sum_tanh_backward_28 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    auto tmp3 = out_ptr3[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<int>(x1);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp2 ? tmp3 : tmp4;
                    out_ptr6[static_cast<long>(x1 + (2L*x0))] = tmp5;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp9 = tmp6 * tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0 + x0_inner)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp9 = out_ptr0[static_cast<long>(x0 + x0_inner)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp13 = out_ptr1[static_cast<long>(x0 + x0_inner)];
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
                        tmp18.store(out_ptr2 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x0) + (2L*x0_inner) + (1536L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_add_mul_pow_sum_tanh_backward_31 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    auto tmp3 = out_ptr3[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<int>(x1);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp2 ? tmp3 : tmp4;
                    out_ptr6[static_cast<long>(x1 + (2L*x0))] = tmp5;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp9 = tmp6 * tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0 + x0_inner)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp9 = out_ptr0[static_cast<long>(x0 + x0_inner)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp13 = out_ptr1[static_cast<long>(x0 + x0_inner)];
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
                        tmp18.store(out_ptr2 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x0) + (2L*x0_inner) + (1536L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_add_mul_pow_sum_tanh_backward_34 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    auto tmp3 = out_ptr3[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<int>(x1);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp2 ? tmp3 : tmp4;
                    out_ptr6[static_cast<long>(x1 + (2L*x0))] = tmp5;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp0, 8);
                        float tmp7[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x1 + (768L*x0)), static_cast<long>(768L), tmp7, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp4 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x1_inner));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 * tmp5;
                            auto tmp9 = tmp6 * tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x0 + x0_inner)];
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp2 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((2L*x1) + (2L*x1_inner) + (1536L*x0) + (1536L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp9 = out_ptr0[static_cast<long>(x0 + x0_inner)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
                        auto tmp13 = out_ptr1[static_cast<long>(x0 + x0_inner)];
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
                        tmp18.store(out_ptr2 + static_cast<long>(x1 + (768L*x0) + (768L*x0_inner)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr1[static_cast<long>((2L*x0) + (2L*x0_inner) + (1536L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_add_mul_pow_sum_tanh_backward_37 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    auto tmp3 = out_ptr3[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<int>(x1);
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 == tmp1;
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = tmp2 ? tmp3 : tmp4;
                    out_ptr6[static_cast<long>(x1 + (2L*x0))] = tmp5;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr0[static_cast<long>(2L*x0)];
                auto tmp3 = in_ptr1[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                auto tmp4 = c10::convert<float>(tmp3);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                auto tmp7 = decltype(tmp2)(tmp2 * tmp6);
                in_out_ptr0[static_cast<long>(x0)] = tmp7;
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_native_layer_norm_backward_nll_loss_forward_sum_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const long* in_ptr5,
                       const long* in_ptr6,
                       const long* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp11 = out_ptr2[static_cast<long>(x0)];
                    auto tmp17 = in_ptr5[static_cast<long>(x0)];
                    auto tmp24 = in_ptr6[static_cast<long>(x0)];
                    auto tmp28 = in_ptr7[static_cast<long>(x0)];
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
                    auto tmp18 = static_cast<int>(-1);
                    auto tmp19 = tmp17 == tmp18;
                    auto tmp20 = static_cast<float>(0.0);
                    auto tmp21 = to_float_mask(tmp19);
                    auto tmp22 = at::vec::Vectorized<float>(tmp20);
                    auto tmp23 = decltype(tmp22)::blendv(tmp16, tmp22, tmp21);
                    auto tmp25 = tmp24 == tmp18;
                    auto tmp26 = to_float_mask(tmp25);
                    auto tmp27 = decltype(tmp22)::blendv(tmp16, tmp22, tmp26);
                    auto tmp29 = static_cast<int>(3);
                    auto tmp30 = tmp28 == tmp29;
                    auto tmp31 = to_float_mask(tmp30);
                    auto tmp32 = decltype(tmp22)::blendv(tmp16, tmp22, tmp31);
                    tmp23.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    tmp27.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    tmp32.store(out_ptr6 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr7 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr8 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr9 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_embedding_dense_backward_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576000L); x0+=static_cast<long>(8L))
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
    primals_4, primals_8, primals_14, primals_16, primals_22, primals_24, primals_30, primals_32, primals_38, primals_40, primals_46, primals_48, primals_54, primals_56, primals_62, primals_64, primals_70, primals_72, primals_78, primals_80, primals_86, primals_88, primals_94, primals_96, primals_102, primals_108, primals_114, primals_115, expand, slice_2, mul, view, getitem_3, mul_2, view_2, addmm_1, tanh, view_4, getitem_7, mul_8, mul_10, view_6, addmm_3, tanh_1, view_8, getitem_13, mul_16, mul_18, view_10, addmm_5, tanh_2, view_12, getitem_19, mul_24, mul_26, view_14, addmm_7, tanh_3, view_16, getitem_25, mul_32, mul_34, view_18, addmm_9, tanh_4, view_20, getitem_31, mul_40, mul_42, view_22, addmm_11, tanh_5, view_24, getitem_37, mul_48, mul_50, view_26, addmm_13, tanh_6, view_28, getitem_43, mul_56, mul_58, view_30, addmm_15, tanh_7, view_32, getitem_49, mul_64, mul_66, view_34, addmm_17, tanh_8, view_36, getitem_55, mul_72, mul_74, view_38, addmm_19, tanh_9, view_40, getitem_61, mul_80, mul_82, view_42, addmm_21, tanh_10, view_44, getitem_67, mul_88, mul_90, view_46, addmm_23, tanh_11, view_48, getitem_73, mul_96, view_50, addmm_26, tanh_13, getitem_77, rsqrt_25, view_52, sub_27, convert_element_type_12, permute_28, permute_32, div_3, permute_36, permute_40, div_4, div_5, permute_44, permute_48, div_6, div_7, permute_52, permute_56, div_8, div_9, permute_60, permute_64, div_10, div_11, permute_68, permute_72, div_12, div_13, permute_76, permute_80, div_14, div_15, permute_84, permute_88, div_16, div_17, permute_92, permute_96, div_18, div_19, permute_100, permute_104, div_20, div_21, permute_108, permute_112, div_22, div_23, permute_116, permute_120, div_24, div_25, permute_124, permute_128, div_26, permute_132, div_27, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_80, (768, ), (1, ))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_96, (768, ), (1, ))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_108, (768, ), (1, ))
    assert_size_stride(primals_114, (1, 512), (512, 1))
    assert_size_stride(primals_115, (1, 512), (512, 1))
    assert_size_stride(expand, (1, 512), (512, 1))
    assert_size_stride(slice_2, (1, 512), (512, 1))
    assert_size_stride(mul, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view, (512, 768), (768, 1))
    assert_size_stride(getitem_3, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_2, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_2, (512, 768), (768, 1))
    assert_size_stride(addmm_1, (512, 3072), (3072, 1))
    assert_size_stride(tanh, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_4, (512, 3072), (3072, 1))
    assert_size_stride(getitem_7, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_8, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_10, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_6, (512, 768), (768, 1))
    assert_size_stride(addmm_3, (512, 3072), (3072, 1))
    assert_size_stride(tanh_1, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_8, (512, 3072), (3072, 1))
    assert_size_stride(getitem_13, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_16, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_18, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_10, (512, 768), (768, 1))
    assert_size_stride(addmm_5, (512, 3072), (3072, 1))
    assert_size_stride(tanh_2, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_12, (512, 3072), (3072, 1))
    assert_size_stride(getitem_19, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_24, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_26, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_14, (512, 768), (768, 1))
    assert_size_stride(addmm_7, (512, 3072), (3072, 1))
    assert_size_stride(tanh_3, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_16, (512, 3072), (3072, 1))
    assert_size_stride(getitem_25, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_32, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_34, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_18, (512, 768), (768, 1))
    assert_size_stride(addmm_9, (512, 3072), (3072, 1))
    assert_size_stride(tanh_4, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_20, (512, 3072), (3072, 1))
    assert_size_stride(getitem_31, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_40, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_42, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_22, (512, 768), (768, 1))
    assert_size_stride(addmm_11, (512, 3072), (3072, 1))
    assert_size_stride(tanh_5, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_24, (512, 3072), (3072, 1))
    assert_size_stride(getitem_37, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_48, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_50, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_26, (512, 768), (768, 1))
    assert_size_stride(addmm_13, (512, 3072), (3072, 1))
    assert_size_stride(tanh_6, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_28, (512, 3072), (3072, 1))
    assert_size_stride(getitem_43, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_56, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_58, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_30, (512, 768), (768, 1))
    assert_size_stride(addmm_15, (512, 3072), (3072, 1))
    assert_size_stride(tanh_7, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_32, (512, 3072), (3072, 1))
    assert_size_stride(getitem_49, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_64, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_66, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_34, (512, 768), (768, 1))
    assert_size_stride(addmm_17, (512, 3072), (3072, 1))
    assert_size_stride(tanh_8, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_36, (512, 3072), (3072, 1))
    assert_size_stride(getitem_55, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_72, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_74, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_38, (512, 768), (768, 1))
    assert_size_stride(addmm_19, (512, 3072), (3072, 1))
    assert_size_stride(tanh_9, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_40, (512, 3072), (3072, 1))
    assert_size_stride(getitem_61, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_80, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_82, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_42, (512, 768), (768, 1))
    assert_size_stride(addmm_21, (512, 3072), (3072, 1))
    assert_size_stride(tanh_10, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_44, (512, 3072), (3072, 1))
    assert_size_stride(getitem_67, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_88, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_90, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_46, (512, 768), (768, 1))
    assert_size_stride(addmm_23, (512, 3072), (3072, 1))
    assert_size_stride(tanh_11, (1, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_48, (512, 3072), (3072, 1))
    assert_size_stride(getitem_73, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_96, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_50, (512, 768), (768, 1))
    assert_size_stride(addmm_26, (512, 768), (768, 1))
    assert_size_stride(tanh_13, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(getitem_77, (1, 512, 1), (512, 1, 1))
    assert_size_stride(rsqrt_25, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_52, (512, 768), (768, 1))
    assert_size_stride(sub_27, (512, 32000), (32000, 1))
    assert_size_stride(convert_element_type_12, (), ())
    assert_size_stride(permute_28, (32000, 768), (768, 1))
    assert_size_stride(permute_32, (768, 768), (768, 1))
    assert_size_stride(div_3, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_36, (768, 3072), (3072, 1))
    assert_size_stride(permute_40, (3072, 768), (768, 1))
    assert_size_stride(div_4, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_5, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_44, (768, 3072), (3072, 1))
    assert_size_stride(permute_48, (3072, 768), (768, 1))
    assert_size_stride(div_6, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_7, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_52, (768, 3072), (3072, 1))
    assert_size_stride(permute_56, (3072, 768), (768, 1))
    assert_size_stride(div_8, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_9, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_60, (768, 3072), (3072, 1))
    assert_size_stride(permute_64, (3072, 768), (768, 1))
    assert_size_stride(div_10, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_11, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_68, (768, 3072), (3072, 1))
    assert_size_stride(permute_72, (3072, 768), (768, 1))
    assert_size_stride(div_12, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_13, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_76, (768, 3072), (3072, 1))
    assert_size_stride(permute_80, (3072, 768), (768, 1))
    assert_size_stride(div_14, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_15, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_84, (768, 3072), (3072, 1))
    assert_size_stride(permute_88, (3072, 768), (768, 1))
    assert_size_stride(div_16, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_17, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_92, (768, 3072), (3072, 1))
    assert_size_stride(permute_96, (3072, 768), (768, 1))
    assert_size_stride(div_18, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_19, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_100, (768, 3072), (3072, 1))
    assert_size_stride(permute_104, (3072, 768), (768, 1))
    assert_size_stride(div_20, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_21, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_108, (768, 3072), (3072, 1))
    assert_size_stride(permute_112, (3072, 768), (768, 1))
    assert_size_stride(div_22, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_23, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_116, (768, 3072), (3072, 1))
    assert_size_stride(permute_120, (3072, 768), (768, 1))
    assert_size_stride(div_24, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_25, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_124, (768, 3072), (3072, 1))
    assert_size_stride(permute_128, (3072, 768), (768, 1))
    assert_size_stride(div_26, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_132, (768, 768), (768, 1))
    assert_size_stride(div_27, (1, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 512, 32000), (16384000, 32000, 1))
    buf0 = empty((512, 32000), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((512, 1), (1, 512), device='cpu', dtype=torch.int64)
    cpp_fused_nll_loss_backward_nll_loss_forward_0(c_void_p(primals_115.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    aten.scatter_(buf0,1,buf1,-1.0)
    del buf1
    buf4 = empty_strided((512, 1), (1, 512), device='cpu', dtype=torch.float32)
    buf3 = empty((512, 32000), device='cpu', dtype=torch.float32)
    buf5 = reinterpret_tensor(buf3, (1, 512, 32000), (16384000, 32000, 1), 0); del buf3  # reuse
    cpp_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_1(c_void_p(buf5.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(convert_element_type_12.data_ptr()), c_void_p(tangents_2.data_ptr()), c_void_p(sub_27.data_ptr()), c_void_p(buf4.data_ptr()))
    del buf0
    del convert_element_type_12
    del primals_115
    del sub_27
    del tangents_1
    del tangents_2
    buf6 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (512, 32000), (32000, 1), 0), permute_28, out=buf6)
    del permute_28
    buf7 = empty((32000, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (32000, 512), (1, 32000), 0), view_52, out=buf7)
    del view_52
    buf8 = empty((1, 32000), device='cpu', dtype=torch.float32)
    buf9 = reinterpret_tensor(buf4, (1, 512, 1), (512, 1, 512), 0); del buf4  # reuse
    buf10 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf11 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf12 = empty((768, ), device='cpu', dtype=torch.float32)
    buf13 = empty((768, ), device='cpu', dtype=torch.float32)
    buf14 = buf11; del buf11  # reuse
    cpp_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_sum_tanh_backward_2(c_void_p(buf14.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(addmm_26.data_ptr()), c_void_p(tanh_13.data_ptr()), c_void_p(getitem_77.data_ptr()), c_void_p(rsqrt_25.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    del addmm_26
    del buf5
    del getitem_77
    del primals_108
    del rsqrt_25
    del tanh_13
    buf15 = buf6; del buf6  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf14, (512, 768), (768, 1), 0), permute_32, out=buf15)
    del permute_32
    buf16 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf14, (768, 512), (1, 768), 0), view_50, out=buf16)
    del view_50
    buf17 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf18 = buf9; del buf9  # reuse
    buf19 = buf10; del buf10  # reuse
    buf20 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf21 = empty((768, ), device='cpu', dtype=torch.float32)
    buf22 = empty((768, ), device='cpu', dtype=torch.float32)
    buf23 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_native_dropout_backward_native_layer_norm_backward_sum_3(c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(mul_96.data_ptr()), c_void_p(div_3.data_ptr()), c_void_p(getitem_73.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()))
    del div_3
    del getitem_73
    del mul_96
    del primals_102
    buf24 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf23, (512, 768), (768, 1), 0), permute_36, out=buf24)
    del permute_36
    buf25 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf23, (768, 512), (1, 768), 0), view_48, out=buf25)
    del view_48
    buf26 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf27 = reinterpret_tensor(buf24, (1, 512, 3072), (1572864, 3072, 1), 0); del buf24  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_4(c_void_p(buf27.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(addmm_23.data_ptr()), c_void_p(tanh_11.data_ptr()), c_void_p(buf26.data_ptr()))
    del addmm_23
    del tanh_11
    buf28 = reinterpret_tensor(buf23, (512, 768), (768, 1), 0); del buf23  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf27, (512, 3072), (3072, 1), 0), permute_40, out=buf28)
    del permute_40
    buf29 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf27, (3072, 512), (1, 3072), 0), view_46, out=buf29)
    del view_46
    buf30 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf31 = buf19; del buf19  # reuse
    buf32 = buf18; del buf18  # reuse
    buf33 = reinterpret_tensor(buf15, (1, 512, 768), (393216, 768, 1), 0); del buf15  # reuse
    buf34 = empty((768, ), device='cpu', dtype=torch.float32)
    buf35 = empty((768, ), device='cpu', dtype=torch.float32)
    buf36 = empty((1, 512, 768, 2), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_5(c_void_p(buf27.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(mul_90.data_ptr()), c_void_p(div_4.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()))
    del div_4
    del mul_90
    del primals_96
    # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
    buf37 = aten.view_as_complex(buf36)
    del buf36
    buf38 = buf37
    del buf37
    # Source Nodes: [], Original ATen: [aten._fft_c2c]
    buf39 = aten._fft_c2c(buf38, [1, 2], 0, False)
    del buf38
    buf40 = buf39
    del buf39
    # Source Nodes: [], Original ATen: [aten.view_as_real]
    buf41 = aten.view_as_real(buf40)
    del buf40
    buf42 = buf41
    del buf41
    buf43 = buf32; del buf32  # reuse
    buf44 = buf31; del buf31  # reuse
    buf45 = reinterpret_tensor(buf28, (1, 512, 768), (393216, 768, 1), 0); del buf28  # reuse
    buf46 = empty((768, ), device='cpu', dtype=torch.float32)
    buf47 = empty((768, ), device='cpu', dtype=torch.float32)
    buf48 = buf20; del buf20  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_6(c_void_p(buf33.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(mul_88.data_ptr()), c_void_p(div_5.data_ptr()), c_void_p(getitem_67.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    del div_5
    del getitem_67
    del mul_88
    del primals_94
    buf49 = reinterpret_tensor(buf27, (512, 3072), (3072, 1), 0); del buf27  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf48, (512, 768), (768, 1), 0), permute_44, out=buf49)
    del permute_44
    buf50 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf48, (768, 512), (1, 768), 0), view_44, out=buf50)
    del view_44
    buf51 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf52 = reinterpret_tensor(buf49, (1, 512, 3072), (1572864, 3072, 1), 0); del buf49  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_7(c_void_p(buf52.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(addmm_21.data_ptr()), c_void_p(tanh_10.data_ptr()), c_void_p(buf51.data_ptr()))
    del addmm_21
    del tanh_10
    buf53 = reinterpret_tensor(buf48, (512, 768), (768, 1), 0); del buf48  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf52, (512, 3072), (3072, 1), 0), permute_48, out=buf53)
    del permute_48
    buf54 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf52, (3072, 512), (1, 3072), 0), view_42, out=buf54)
    del view_42
    buf55 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf56 = buf44; del buf44  # reuse
    buf57 = buf43; del buf43  # reuse
    buf58 = buf33; del buf33  # reuse
    buf59 = empty((768, ), device='cpu', dtype=torch.float32)
    buf60 = empty((768, ), device='cpu', dtype=torch.float32)
    buf61 = buf42; del buf42  # reuse
    cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_8(c_void_p(buf52.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(mul_82.data_ptr()), c_void_p(div_6.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()))
    del div_6
    del mul_82
    del primals_88
    # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
    buf62 = aten.view_as_complex(buf61)
    del buf61
    buf63 = buf62
    del buf62
    # Source Nodes: [], Original ATen: [aten._fft_c2c]
    buf64 = aten._fft_c2c(buf63, [1, 2], 0, False)
    del buf63
    buf65 = buf64
    del buf64
    # Source Nodes: [], Original ATen: [aten.view_as_real]
    buf66 = aten.view_as_real(buf65)
    del buf65
    buf67 = buf66
    del buf66
    buf68 = buf57; del buf57  # reuse
    buf69 = buf56; del buf56  # reuse
    buf70 = reinterpret_tensor(buf53, (1, 512, 768), (393216, 768, 1), 0); del buf53  # reuse
    buf71 = empty((768, ), device='cpu', dtype=torch.float32)
    buf72 = empty((768, ), device='cpu', dtype=torch.float32)
    buf73 = buf45; del buf45  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_9(c_void_p(buf58.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(mul_80.data_ptr()), c_void_p(div_7.data_ptr()), c_void_p(getitem_61.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()))
    del div_7
    del getitem_61
    del mul_80
    del primals_86
    buf74 = reinterpret_tensor(buf52, (512, 3072), (3072, 1), 0); del buf52  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf73, (512, 768), (768, 1), 0), permute_52, out=buf74)
    del permute_52
    buf75 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf73, (768, 512), (1, 768), 0), view_40, out=buf75)
    del view_40
    buf76 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf77 = reinterpret_tensor(buf74, (1, 512, 3072), (1572864, 3072, 1), 0); del buf74  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_10(c_void_p(buf77.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(addmm_19.data_ptr()), c_void_p(tanh_9.data_ptr()), c_void_p(buf76.data_ptr()))
    del addmm_19
    del tanh_9
    buf78 = reinterpret_tensor(buf73, (512, 768), (768, 1), 0); del buf73  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf77, (512, 3072), (3072, 1), 0), permute_56, out=buf78)
    del permute_56
    buf79 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf77, (3072, 512), (1, 3072), 0), view_38, out=buf79)
    del view_38
    buf80 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf81 = buf69; del buf69  # reuse
    buf82 = buf68; del buf68  # reuse
    buf83 = buf58; del buf58  # reuse
    buf84 = empty((768, ), device='cpu', dtype=torch.float32)
    buf85 = empty((768, ), device='cpu', dtype=torch.float32)
    buf86 = buf67; del buf67  # reuse
    cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_11(c_void_p(buf77.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(mul_74.data_ptr()), c_void_p(div_8.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()))
    del div_8
    del mul_74
    del primals_80
    # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
    buf87 = aten.view_as_complex(buf86)
    del buf86
    buf88 = buf87
    del buf87
    # Source Nodes: [], Original ATen: [aten._fft_c2c]
    buf89 = aten._fft_c2c(buf88, [1, 2], 0, False)
    del buf88
    buf90 = buf89
    del buf89
    # Source Nodes: [], Original ATen: [aten.view_as_real]
    buf91 = aten.view_as_real(buf90)
    del buf90
    buf92 = buf91
    del buf91
    buf93 = buf82; del buf82  # reuse
    buf94 = buf81; del buf81  # reuse
    buf95 = reinterpret_tensor(buf78, (1, 512, 768), (393216, 768, 1), 0); del buf78  # reuse
    buf96 = empty((768, ), device='cpu', dtype=torch.float32)
    buf97 = empty((768, ), device='cpu', dtype=torch.float32)
    buf98 = buf70; del buf70  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_12(c_void_p(buf83.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(mul_72.data_ptr()), c_void_p(div_9.data_ptr()), c_void_p(getitem_55.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()))
    del div_9
    del getitem_55
    del mul_72
    del primals_78
    buf99 = reinterpret_tensor(buf77, (512, 3072), (3072, 1), 0); del buf77  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf98, (512, 768), (768, 1), 0), permute_60, out=buf99)
    del permute_60
    buf100 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf98, (768, 512), (1, 768), 0), view_36, out=buf100)
    del view_36
    buf101 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf102 = reinterpret_tensor(buf99, (1, 512, 3072), (1572864, 3072, 1), 0); del buf99  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_13(c_void_p(buf102.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(addmm_17.data_ptr()), c_void_p(tanh_8.data_ptr()), c_void_p(buf101.data_ptr()))
    del addmm_17
    del tanh_8
    buf103 = reinterpret_tensor(buf98, (512, 768), (768, 1), 0); del buf98  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf102, (512, 3072), (3072, 1), 0), permute_64, out=buf103)
    del permute_64
    buf104 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf102, (3072, 512), (1, 3072), 0), view_34, out=buf104)
    del view_34
    buf105 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf106 = buf94; del buf94  # reuse
    buf107 = buf93; del buf93  # reuse
    buf108 = buf83; del buf83  # reuse
    buf109 = empty((768, ), device='cpu', dtype=torch.float32)
    buf110 = empty((768, ), device='cpu', dtype=torch.float32)
    buf111 = buf92; del buf92  # reuse
    cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_14(c_void_p(buf102.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(mul_66.data_ptr()), c_void_p(div_10.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()))
    del div_10
    del mul_66
    del primals_72
    # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
    buf112 = aten.view_as_complex(buf111)
    del buf111
    buf113 = buf112
    del buf112
    # Source Nodes: [], Original ATen: [aten._fft_c2c]
    buf114 = aten._fft_c2c(buf113, [1, 2], 0, False)
    del buf113
    buf115 = buf114
    del buf114
    # Source Nodes: [], Original ATen: [aten.view_as_real]
    buf116 = aten.view_as_real(buf115)
    del buf115
    buf117 = buf116
    del buf116
    buf118 = buf107; del buf107  # reuse
    buf119 = buf106; del buf106  # reuse
    buf120 = buf95; del buf95  # reuse
    buf121 = empty((768, ), device='cpu', dtype=torch.float32)
    buf122 = empty((768, ), device='cpu', dtype=torch.float32)
    buf123 = reinterpret_tensor(buf103, (1, 512, 768), (393216, 768, 1), 0); del buf103  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_15(c_void_p(buf108.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(mul_64.data_ptr()), c_void_p(div_11.data_ptr()), c_void_p(getitem_49.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()))
    del div_11
    del getitem_49
    del mul_64
    del primals_70
    buf124 = reinterpret_tensor(buf102, (512, 3072), (3072, 1), 0); del buf102  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf123, (512, 768), (768, 1), 0), permute_68, out=buf124)
    del permute_68
    buf125 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf123, (768, 512), (1, 768), 0), view_32, out=buf125)
    del view_32
    buf126 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf127 = reinterpret_tensor(buf124, (1, 512, 3072), (1572864, 3072, 1), 0); del buf124  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_16(c_void_p(buf127.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(addmm_15.data_ptr()), c_void_p(tanh_7.data_ptr()), c_void_p(buf126.data_ptr()))
    del addmm_15
    del tanh_7
    buf128 = reinterpret_tensor(buf123, (512, 768), (768, 1), 0); del buf123  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf127, (512, 3072), (3072, 1), 0), permute_72, out=buf128)
    del permute_72
    buf129 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf127, (3072, 512), (1, 3072), 0), view_30, out=buf129)
    del view_30
    buf130 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf131 = buf119; del buf119  # reuse
    buf132 = buf118; del buf118  # reuse
    buf133 = buf108; del buf108  # reuse
    buf134 = empty((768, ), device='cpu', dtype=torch.float32)
    buf135 = empty((768, ), device='cpu', dtype=torch.float32)
    buf136 = buf117; del buf117  # reuse
    cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_17(c_void_p(buf127.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(mul_58.data_ptr()), c_void_p(div_12.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()))
    del div_12
    del mul_58
    del primals_64
    # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
    buf137 = aten.view_as_complex(buf136)
    del buf136
    buf138 = buf137
    del buf137
    # Source Nodes: [], Original ATen: [aten._fft_c2c]
    buf139 = aten._fft_c2c(buf138, [1, 2], 0, False)
    del buf138
    buf140 = buf139
    del buf139
    # Source Nodes: [], Original ATen: [aten.view_as_real]
    buf141 = aten.view_as_real(buf140)
    del buf140
    buf142 = buf141
    del buf141
    buf143 = buf132; del buf132  # reuse
    buf144 = buf131; del buf131  # reuse
    buf145 = reinterpret_tensor(buf128, (1, 512, 768), (393216, 768, 1), 0); del buf128  # reuse
    buf146 = empty((768, ), device='cpu', dtype=torch.float32)
    buf147 = empty((768, ), device='cpu', dtype=torch.float32)
    buf148 = buf120; del buf120  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_18(c_void_p(buf133.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(mul_56.data_ptr()), c_void_p(div_13.data_ptr()), c_void_p(getitem_43.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()))
    del div_13
    del getitem_43
    del mul_56
    del primals_62
    buf149 = reinterpret_tensor(buf127, (512, 3072), (3072, 1), 0); del buf127  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf148, (512, 768), (768, 1), 0), permute_76, out=buf149)
    del permute_76
    buf150 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf148, (768, 512), (1, 768), 0), view_28, out=buf150)
    del view_28
    buf151 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf152 = reinterpret_tensor(buf149, (1, 512, 3072), (1572864, 3072, 1), 0); del buf149  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_19(c_void_p(buf152.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(addmm_13.data_ptr()), c_void_p(tanh_6.data_ptr()), c_void_p(buf151.data_ptr()))
    del addmm_13
    del tanh_6
    buf153 = reinterpret_tensor(buf148, (512, 768), (768, 1), 0); del buf148  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf152, (512, 3072), (3072, 1), 0), permute_80, out=buf153)
    del permute_80
    buf154 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf152, (3072, 512), (1, 3072), 0), view_26, out=buf154)
    del view_26
    buf155 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf156 = buf144; del buf144  # reuse
    buf157 = buf143; del buf143  # reuse
    buf158 = buf133; del buf133  # reuse
    buf159 = empty((768, ), device='cpu', dtype=torch.float32)
    buf160 = empty((768, ), device='cpu', dtype=torch.float32)
    buf161 = buf142; del buf142  # reuse
    cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_20(c_void_p(buf152.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(mul_50.data_ptr()), c_void_p(div_14.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()))
    del div_14
    del mul_50
    del primals_56
    # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
    buf162 = aten.view_as_complex(buf161)
    del buf161
    buf163 = buf162
    del buf162
    # Source Nodes: [], Original ATen: [aten._fft_c2c]
    buf164 = aten._fft_c2c(buf163, [1, 2], 0, False)
    del buf163
    buf165 = buf164
    del buf164
    # Source Nodes: [], Original ATen: [aten.view_as_real]
    buf166 = aten.view_as_real(buf165)
    del buf165
    buf167 = buf166
    del buf166
    buf168 = buf157; del buf157  # reuse
    buf169 = buf156; del buf156  # reuse
    buf170 = reinterpret_tensor(buf153, (1, 512, 768), (393216, 768, 1), 0); del buf153  # reuse
    buf171 = empty((768, ), device='cpu', dtype=torch.float32)
    buf172 = empty((768, ), device='cpu', dtype=torch.float32)
    buf173 = buf145; del buf145  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_21(c_void_p(buf158.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(mul_48.data_ptr()), c_void_p(div_15.data_ptr()), c_void_p(getitem_37.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()))
    del div_15
    del getitem_37
    del mul_48
    del primals_54
    buf174 = reinterpret_tensor(buf152, (512, 3072), (3072, 1), 0); del buf152  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf173, (512, 768), (768, 1), 0), permute_84, out=buf174)
    del permute_84
    buf175 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf173, (768, 512), (1, 768), 0), view_24, out=buf175)
    del view_24
    buf176 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf177 = reinterpret_tensor(buf174, (1, 512, 3072), (1572864, 3072, 1), 0); del buf174  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_22(c_void_p(buf177.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(addmm_11.data_ptr()), c_void_p(tanh_5.data_ptr()), c_void_p(buf176.data_ptr()))
    del addmm_11
    del tanh_5
    buf178 = reinterpret_tensor(buf173, (512, 768), (768, 1), 0); del buf173  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf177, (512, 3072), (3072, 1), 0), permute_88, out=buf178)
    del permute_88
    buf179 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf177, (3072, 512), (1, 3072), 0), view_22, out=buf179)
    del view_22
    buf180 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf181 = buf169; del buf169  # reuse
    buf182 = buf168; del buf168  # reuse
    buf183 = buf158; del buf158  # reuse
    buf184 = empty((768, ), device='cpu', dtype=torch.float32)
    buf185 = empty((768, ), device='cpu', dtype=torch.float32)
    buf186 = buf167; del buf167  # reuse
    cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_23(c_void_p(buf177.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(mul_42.data_ptr()), c_void_p(div_16.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()))
    del div_16
    del mul_42
    del primals_48
    # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
    buf187 = aten.view_as_complex(buf186)
    del buf186
    buf188 = buf187
    del buf187
    # Source Nodes: [], Original ATen: [aten._fft_c2c]
    buf189 = aten._fft_c2c(buf188, [1, 2], 0, False)
    del buf188
    buf190 = buf189
    del buf189
    # Source Nodes: [], Original ATen: [aten.view_as_real]
    buf191 = aten.view_as_real(buf190)
    del buf190
    buf192 = buf191
    del buf191
    buf193 = buf182; del buf182  # reuse
    buf194 = buf181; del buf181  # reuse
    buf195 = reinterpret_tensor(buf178, (1, 512, 768), (393216, 768, 1), 0); del buf178  # reuse
    buf196 = empty((768, ), device='cpu', dtype=torch.float32)
    buf197 = empty((768, ), device='cpu', dtype=torch.float32)
    buf198 = buf170; del buf170  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_24(c_void_p(buf183.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(mul_40.data_ptr()), c_void_p(div_17.data_ptr()), c_void_p(getitem_31.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()))
    del div_17
    del getitem_31
    del mul_40
    del primals_46
    buf199 = reinterpret_tensor(buf177, (512, 3072), (3072, 1), 0); del buf177  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf198, (512, 768), (768, 1), 0), permute_92, out=buf199)
    del permute_92
    buf200 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf198, (768, 512), (1, 768), 0), view_20, out=buf200)
    del view_20
    buf201 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf202 = reinterpret_tensor(buf199, (1, 512, 3072), (1572864, 3072, 1), 0); del buf199  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_25(c_void_p(buf202.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(addmm_9.data_ptr()), c_void_p(tanh_4.data_ptr()), c_void_p(buf201.data_ptr()))
    del addmm_9
    del tanh_4
    buf203 = reinterpret_tensor(buf198, (512, 768), (768, 1), 0); del buf198  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf202, (512, 3072), (3072, 1), 0), permute_96, out=buf203)
    del permute_96
    buf204 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf202, (3072, 512), (1, 3072), 0), view_18, out=buf204)
    del view_18
    buf205 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf206 = buf194; del buf194  # reuse
    buf207 = buf193; del buf193  # reuse
    buf208 = buf183; del buf183  # reuse
    buf209 = empty((768, ), device='cpu', dtype=torch.float32)
    buf210 = empty((768, ), device='cpu', dtype=torch.float32)
    buf211 = buf192; del buf192  # reuse
    cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_26(c_void_p(buf202.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(mul_34.data_ptr()), c_void_p(div_18.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()))
    del div_18
    del mul_34
    del primals_40
    # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
    buf212 = aten.view_as_complex(buf211)
    del buf211
    buf213 = buf212
    del buf212
    # Source Nodes: [], Original ATen: [aten._fft_c2c]
    buf214 = aten._fft_c2c(buf213, [1, 2], 0, False)
    del buf213
    buf215 = buf214
    del buf214
    # Source Nodes: [], Original ATen: [aten.view_as_real]
    buf216 = aten.view_as_real(buf215)
    del buf215
    buf217 = buf216
    del buf216
    buf218 = buf207; del buf207  # reuse
    buf219 = buf206; del buf206  # reuse
    buf220 = reinterpret_tensor(buf203, (1, 512, 768), (393216, 768, 1), 0); del buf203  # reuse
    buf221 = empty((768, ), device='cpu', dtype=torch.float32)
    buf222 = empty((768, ), device='cpu', dtype=torch.float32)
    buf223 = buf195; del buf195  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_27(c_void_p(buf208.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(mul_32.data_ptr()), c_void_p(div_19.data_ptr()), c_void_p(getitem_25.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()))
    del div_19
    del getitem_25
    del mul_32
    del primals_38
    buf224 = reinterpret_tensor(buf202, (512, 3072), (3072, 1), 0); del buf202  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf223, (512, 768), (768, 1), 0), permute_100, out=buf224)
    del permute_100
    buf225 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf223, (768, 512), (1, 768), 0), view_16, out=buf225)
    del view_16
    buf226 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf227 = reinterpret_tensor(buf224, (1, 512, 3072), (1572864, 3072, 1), 0); del buf224  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_28(c_void_p(buf227.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(addmm_7.data_ptr()), c_void_p(tanh_3.data_ptr()), c_void_p(buf226.data_ptr()))
    del addmm_7
    del tanh_3
    buf228 = reinterpret_tensor(buf223, (512, 768), (768, 1), 0); del buf223  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf227, (512, 3072), (3072, 1), 0), permute_104, out=buf228)
    del permute_104
    buf229 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf227, (3072, 512), (1, 3072), 0), view_14, out=buf229)
    del view_14
    buf230 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf231 = buf219; del buf219  # reuse
    buf232 = buf218; del buf218  # reuse
    buf233 = buf208; del buf208  # reuse
    buf234 = empty((768, ), device='cpu', dtype=torch.float32)
    buf235 = empty((768, ), device='cpu', dtype=torch.float32)
    buf236 = buf217; del buf217  # reuse
    cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_29(c_void_p(buf227.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(mul_26.data_ptr()), c_void_p(div_20.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()))
    del div_20
    del mul_26
    del primals_32
    # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
    buf237 = aten.view_as_complex(buf236)
    del buf236
    buf238 = buf237
    del buf237
    # Source Nodes: [], Original ATen: [aten._fft_c2c]
    buf239 = aten._fft_c2c(buf238, [1, 2], 0, False)
    del buf238
    buf240 = buf239
    del buf239
    # Source Nodes: [], Original ATen: [aten.view_as_real]
    buf241 = aten.view_as_real(buf240)
    del buf240
    buf242 = buf241
    del buf241
    buf243 = buf232; del buf232  # reuse
    buf244 = buf231; del buf231  # reuse
    buf245 = reinterpret_tensor(buf228, (1, 512, 768), (393216, 768, 1), 0); del buf228  # reuse
    buf246 = empty((768, ), device='cpu', dtype=torch.float32)
    buf247 = empty((768, ), device='cpu', dtype=torch.float32)
    buf248 = buf220; del buf220  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_30(c_void_p(buf233.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(mul_24.data_ptr()), c_void_p(div_21.data_ptr()), c_void_p(getitem_19.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()))
    del div_21
    del getitem_19
    del mul_24
    del primals_30
    buf249 = reinterpret_tensor(buf227, (512, 3072), (3072, 1), 0); del buf227  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf248, (512, 768), (768, 1), 0), permute_108, out=buf249)
    del permute_108
    buf250 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf248, (768, 512), (1, 768), 0), view_12, out=buf250)
    del view_12
    buf251 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf252 = reinterpret_tensor(buf249, (1, 512, 3072), (1572864, 3072, 1), 0); del buf249  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_31(c_void_p(buf252.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(addmm_5.data_ptr()), c_void_p(tanh_2.data_ptr()), c_void_p(buf251.data_ptr()))
    del addmm_5
    del tanh_2
    buf253 = reinterpret_tensor(buf248, (512, 768), (768, 1), 0); del buf248  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf252, (512, 3072), (3072, 1), 0), permute_112, out=buf253)
    del permute_112
    buf254 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf252, (3072, 512), (1, 3072), 0), view_10, out=buf254)
    del view_10
    buf255 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf256 = buf244; del buf244  # reuse
    buf257 = buf243; del buf243  # reuse
    buf258 = buf233; del buf233  # reuse
    buf259 = empty((768, ), device='cpu', dtype=torch.float32)
    buf260 = empty((768, ), device='cpu', dtype=torch.float32)
    buf261 = buf242; del buf242  # reuse
    cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_32(c_void_p(buf252.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(mul_18.data_ptr()), c_void_p(div_22.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()))
    del div_22
    del mul_18
    del primals_24
    # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
    buf262 = aten.view_as_complex(buf261)
    del buf261
    buf263 = buf262
    del buf262
    # Source Nodes: [], Original ATen: [aten._fft_c2c]
    buf264 = aten._fft_c2c(buf263, [1, 2], 0, False)
    del buf263
    buf265 = buf264
    del buf264
    # Source Nodes: [], Original ATen: [aten.view_as_real]
    buf266 = aten.view_as_real(buf265)
    del buf265
    buf267 = buf266
    del buf266
    buf268 = buf257; del buf257  # reuse
    buf269 = buf256; del buf256  # reuse
    buf270 = reinterpret_tensor(buf253, (1, 512, 768), (393216, 768, 1), 0); del buf253  # reuse
    buf271 = empty((768, ), device='cpu', dtype=torch.float32)
    buf272 = empty((768, ), device='cpu', dtype=torch.float32)
    buf273 = buf245; del buf245  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_33(c_void_p(buf258.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_16.data_ptr()), c_void_p(div_23.data_ptr()), c_void_p(getitem_13.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()))
    del div_23
    del getitem_13
    del mul_16
    del primals_22
    buf274 = reinterpret_tensor(buf252, (512, 3072), (3072, 1), 0); del buf252  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf273, (512, 768), (768, 1), 0), permute_116, out=buf274)
    del permute_116
    buf275 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf273, (768, 512), (1, 768), 0), view_8, out=buf275)
    del view_8
    buf276 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf277 = reinterpret_tensor(buf274, (1, 512, 3072), (1572864, 3072, 1), 0); del buf274  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_34(c_void_p(buf277.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(addmm_3.data_ptr()), c_void_p(tanh_1.data_ptr()), c_void_p(buf276.data_ptr()))
    del addmm_3
    del tanh_1
    buf278 = reinterpret_tensor(buf273, (512, 768), (768, 1), 0); del buf273  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf277, (512, 3072), (3072, 1), 0), permute_120, out=buf278)
    del permute_120
    buf279 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf277, (3072, 512), (1, 3072), 0), view_6, out=buf279)
    del view_6
    buf280 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf281 = buf269; del buf269  # reuse
    buf282 = buf268; del buf268  # reuse
    buf283 = buf258; del buf258  # reuse
    buf284 = empty((768, ), device='cpu', dtype=torch.float32)
    buf285 = empty((768, ), device='cpu', dtype=torch.float32)
    buf286 = buf267; del buf267  # reuse
    cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_35(c_void_p(buf277.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_10.data_ptr()), c_void_p(div_24.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()))
    del div_24
    del mul_10
    del primals_16
    # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
    buf287 = aten.view_as_complex(buf286)
    del buf286
    buf288 = buf287
    del buf287
    # Source Nodes: [], Original ATen: [aten._fft_c2c]
    buf289 = aten._fft_c2c(buf288, [1, 2], 0, False)
    del buf288
    buf290 = buf289
    del buf289
    # Source Nodes: [], Original ATen: [aten.view_as_real]
    buf291 = aten.view_as_real(buf290)
    del buf290
    buf292 = buf291
    del buf291
    buf293 = buf282; del buf282  # reuse
    buf294 = buf281; del buf281  # reuse
    buf295 = reinterpret_tensor(buf278, (1, 512, 768), (393216, 768, 1), 0); del buf278  # reuse
    buf296 = empty((768, ), device='cpu', dtype=torch.float32)
    buf297 = empty((768, ), device='cpu', dtype=torch.float32)
    buf298 = buf270; del buf270  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_36(c_void_p(buf283.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(mul_8.data_ptr()), c_void_p(div_25.data_ptr()), c_void_p(getitem_7.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()))
    del div_25
    del getitem_7
    del mul_8
    del primals_14
    buf299 = reinterpret_tensor(buf277, (512, 3072), (3072, 1), 0); del buf277  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf298, (512, 768), (768, 1), 0), permute_124, out=buf299)
    del permute_124
    buf300 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf298, (768, 512), (1, 768), 0), view_4, out=buf300)
    del view_4
    buf301 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf302 = reinterpret_tensor(buf299, (1, 512, 3072), (1572864, 3072, 1), 0); del buf299  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_37(c_void_p(buf302.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(addmm_1.data_ptr()), c_void_p(tanh.data_ptr()), c_void_p(buf301.data_ptr()))
    del addmm_1
    del tanh
    buf303 = reinterpret_tensor(buf298, (512, 768), (768, 1), 0); del buf298  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf302, (512, 3072), (3072, 1), 0), permute_128, out=buf303)
    del permute_128
    buf304 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf302, (3072, 512), (1, 3072), 0), view_2, out=buf304)
    del view_2
    buf305 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf306 = buf294; del buf294  # reuse
    buf307 = buf293; del buf293  # reuse
    buf308 = buf283; del buf283  # reuse
    buf309 = empty((768, ), device='cpu', dtype=torch.float32)
    buf310 = empty((768, ), device='cpu', dtype=torch.float32)
    buf311 = buf292; del buf292  # reuse
    cpp_fused_add_native_layer_norm_backward_select_backward_sum_view_as_complex_38(c_void_p(buf302.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(mul_2.data_ptr()), c_void_p(div_26.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()))
    del buf302
    del div_26
    del mul_2
    del primals_8
    # Source Nodes: [], Original ATen: [aten.select_backward, aten.view_as_complex]
    buf312 = aten.view_as_complex(buf311)
    del buf311
    buf313 = buf312
    del buf312
    # Source Nodes: [], Original ATen: [aten._fft_c2c]
    buf314 = aten._fft_c2c(buf313, [1, 2], 0, False)
    del buf313
    buf315 = buf314
    del buf314
    # Source Nodes: [], Original ATen: [aten.view_as_real]
    buf316 = aten.view_as_real(buf315)
    del buf315
    buf317 = buf316
    del buf316
    buf318 = buf308; del buf308  # reuse
    cpp_fused_add_native_dropout_backward_39(c_void_p(buf318.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(getitem_3.data_ptr()))
    del buf317
    del getitem_3
    buf319 = buf303; del buf303  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf318, (512, 768), (768, 1), 0), permute_132, out=buf319)
    del permute_132
    buf320 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf318, (768, 512), (1, 768), 0), view, out=buf320)
    del view
    buf321 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf322 = buf307; del buf307  # reuse
    buf323 = buf306; del buf306  # reuse
    buf328 = buf295; del buf295  # reuse
    buf332 = buf14; del buf14  # reuse
    buf336 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf325 = empty((768, ), device='cpu', dtype=torch.float32)
    buf326 = empty((768, ), device='cpu', dtype=torch.float32)
    buf327 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_native_layer_norm_backward_nll_loss_forward_sum_40(c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(mul.data_ptr()), c_void_p(div_27.data_ptr()), c_void_p(slice_2.data_ptr()), c_void_p(expand.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()))
    del buf318
    del buf319
    del buf322
    del buf323
    del div_27
    del mul
    del primals_4
    aten.index_put_(buf327, [slice_2], buf328, True)
    del buf328
    del slice_2
    buf331 = empty((4, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_41(c_void_p(buf331.data_ptr()))
    aten.index_put_(buf331, [expand], buf332, True)
    del buf332
    del expand
    buf335 = empty((32000, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_42(c_void_p(buf335.data_ptr()))
    aten.index_put_(buf335, [primals_114], buf336, True)
    del buf336
    del primals_114
    return (buf335, buf331, buf327, buf325, buf326, reinterpret_tensor(buf320, (768, 768), (768, 1), 0), reinterpret_tensor(buf321, (768, ), (1, ), 0), buf309, buf310, reinterpret_tensor(buf304, (3072, 768), (768, 1), 0), reinterpret_tensor(buf305, (3072, ), (1, ), 0), reinterpret_tensor(buf300, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf301, (768, ), (1, ), 0), buf296, buf297, buf284, buf285, reinterpret_tensor(buf279, (3072, 768), (768, 1), 0), reinterpret_tensor(buf280, (3072, ), (1, ), 0), reinterpret_tensor(buf275, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf276, (768, ), (1, ), 0), buf271, buf272, buf259, buf260, reinterpret_tensor(buf254, (3072, 768), (768, 1), 0), reinterpret_tensor(buf255, (3072, ), (1, ), 0), reinterpret_tensor(buf250, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf251, (768, ), (1, ), 0), buf246, buf247, buf234, buf235, reinterpret_tensor(buf229, (3072, 768), (768, 1), 0), reinterpret_tensor(buf230, (3072, ), (1, ), 0), reinterpret_tensor(buf225, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf226, (768, ), (1, ), 0), buf221, buf222, buf209, buf210, reinterpret_tensor(buf204, (3072, 768), (768, 1), 0), reinterpret_tensor(buf205, (3072, ), (1, ), 0), reinterpret_tensor(buf200, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf201, (768, ), (1, ), 0), buf196, buf197, buf184, buf185, reinterpret_tensor(buf179, (3072, 768), (768, 1), 0), reinterpret_tensor(buf180, (3072, ), (1, ), 0), reinterpret_tensor(buf175, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf176, (768, ), (1, ), 0), buf171, buf172, buf159, buf160, reinterpret_tensor(buf154, (3072, 768), (768, 1), 0), reinterpret_tensor(buf155, (3072, ), (1, ), 0), reinterpret_tensor(buf150, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf151, (768, ), (1, ), 0), buf146, buf147, buf134, buf135, reinterpret_tensor(buf129, (3072, 768), (768, 1), 0), reinterpret_tensor(buf130, (3072, ), (1, ), 0), reinterpret_tensor(buf125, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf126, (768, ), (1, ), 0), buf121, buf122, buf109, buf110, reinterpret_tensor(buf104, (3072, 768), (768, 1), 0), reinterpret_tensor(buf105, (3072, ), (1, ), 0), reinterpret_tensor(buf100, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf101, (768, ), (1, ), 0), buf96, buf97, buf84, buf85, reinterpret_tensor(buf79, (3072, 768), (768, 1), 0), reinterpret_tensor(buf80, (3072, ), (1, ), 0), reinterpret_tensor(buf75, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf76, (768, ), (1, ), 0), buf71, buf72, buf59, buf60, reinterpret_tensor(buf54, (3072, 768), (768, 1), 0), reinterpret_tensor(buf55, (3072, ), (1, ), 0), reinterpret_tensor(buf50, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf51, (768, ), (1, ), 0), buf46, buf47, buf34, buf35, reinterpret_tensor(buf29, (3072, 768), (768, 1), 0), reinterpret_tensor(buf30, (3072, ), (1, ), 0), reinterpret_tensor(buf25, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf26, (768, ), (1, ), 0), buf21, buf22, None, None, reinterpret_tensor(buf16, (768, 768), (768, 1), 0), reinterpret_tensor(buf17, (768, ), (1, ), 0), buf12, buf13, reinterpret_tensor(buf7, (32000, 768), (768, 1), 0), reinterpret_tensor(buf8, (32000, ), (1, ), 0), None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    primals_115 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    expand = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    slice_2 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    mul = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_2 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_2 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_1 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_4 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_8 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    mul_10 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_6 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_3 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_1 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_8 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_13 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_16 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    mul_18 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_10 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_5 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_2 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_12 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_19 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_24 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    mul_26 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_14 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_7 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_3 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_16 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_25 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_32 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    mul_34 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_18 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_9 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_4 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_20 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_31 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_40 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    mul_42 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_22 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_11 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_5 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_24 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_37 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_48 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    mul_50 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_26 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_13 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_6 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_28 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_43 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_56 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    mul_58 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_30 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_15 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_7 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_32 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_49 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_64 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    mul_66 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_34 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_17 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_8 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_36 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_55 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_72 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    mul_74 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_38 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_19 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_9 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_40 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_61 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_80 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    mul_82 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_42 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_21 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_10 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_44 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_67 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_88 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    mul_90 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_46 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_23 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_11 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_48 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_73 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_96 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_50 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_26 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    tanh_13 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    getitem_77 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_25 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_52 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    sub_27 = rand_strided((512, 32000), (32000, 1), device='cpu', dtype=torch.float32)
    convert_element_type_12 = rand_strided((), (), device='cpu', dtype=torch.float32)
    permute_28 = rand_strided((32000, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_32 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_3 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_36 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_40 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_4 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_5 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_44 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_48 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_6 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_7 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_52 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_56 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_8 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_9 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_60 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_64 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_10 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_11 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_68 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_72 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_12 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_13 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_76 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_80 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_14 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_15 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_84 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_88 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_16 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_17 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_92 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_96 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_18 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_19 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_100 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_104 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_20 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_21 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_108 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_112 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_22 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_23 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_116 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_120 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_25 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_124 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_128 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_26 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_132 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_27 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512, 32000), (16384000, 32000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_8, primals_14, primals_16, primals_22, primals_24, primals_30, primals_32, primals_38, primals_40, primals_46, primals_48, primals_54, primals_56, primals_62, primals_64, primals_70, primals_72, primals_78, primals_80, primals_86, primals_88, primals_94, primals_96, primals_102, primals_108, primals_114, primals_115, expand, slice_2, mul, view, getitem_3, mul_2, view_2, addmm_1, tanh, view_4, getitem_7, mul_8, mul_10, view_6, addmm_3, tanh_1, view_8, getitem_13, mul_16, mul_18, view_10, addmm_5, tanh_2, view_12, getitem_19, mul_24, mul_26, view_14, addmm_7, tanh_3, view_16, getitem_25, mul_32, mul_34, view_18, addmm_9, tanh_4, view_20, getitem_31, mul_40, mul_42, view_22, addmm_11, tanh_5, view_24, getitem_37, mul_48, mul_50, view_26, addmm_13, tanh_6, view_28, getitem_43, mul_56, mul_58, view_30, addmm_15, tanh_7, view_32, getitem_49, mul_64, mul_66, view_34, addmm_17, tanh_8, view_36, getitem_55, mul_72, mul_74, view_38, addmm_19, tanh_9, view_40, getitem_61, mul_80, mul_82, view_42, addmm_21, tanh_10, view_44, getitem_67, mul_88, mul_90, view_46, addmm_23, tanh_11, view_48, getitem_73, mul_96, view_50, addmm_26, tanh_13, getitem_77, rsqrt_25, view_52, sub_27, convert_element_type_12, permute_28, permute_32, div_3, permute_36, permute_40, div_4, div_5, permute_44, permute_48, div_6, div_7, permute_52, permute_56, div_8, div_9, permute_60, permute_64, div_10, div_11, permute_68, permute_72, div_12, div_13, permute_76, permute_80, div_14, div_15, permute_84, permute_88, div_16, div_17, permute_92, permute_96, div_18, div_19, permute_100, permute_104, div_20, div_21, permute_108, permute_112, div_22, div_23, permute_116, permute_120, div_24, div_25, permute_124, permute_128, div_26, permute_132, div_27, tangents_1, tangents_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('GoogleFnet', benchmark_compiled_module)
