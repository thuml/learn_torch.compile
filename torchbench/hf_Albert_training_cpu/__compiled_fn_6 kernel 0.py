
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


cpp_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_sum_tanh_backward_0 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(30000L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (30000L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp20 = in_ptr5[static_cast<long>(x0)];
                    auto tmp25 = out_ptr2[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(128.0);
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
                    tmp30.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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


cpp_fused_native_layer_norm_backward_sum_1 = async_compile.cpp('''
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
                       float* out_ptr5)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_add_native_layer_norm_backward_sum_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp2;
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


cpp_fused_add_native_layer_norm_backward_sum_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(8.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((512L*x1) + (512L*x1_inner) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>(x0) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                        tmp_acc2_vec = tmp_acc2_vec + tmp6;
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


cpp_fused_sum_11 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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


cpp_fused_sum_12 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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


cpp_fused_add_native_layer_norm_backward_sum_13 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr2[static_cast<long>(x0)];
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
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_add_native_layer_norm_backward_sum_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp2;
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


cpp_fused_add_native_layer_norm_backward_sum_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(8.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((512L*x1) + (512L*x1_inner) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>(x0) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                        tmp_acc2_vec = tmp_acc2_vec + tmp6;
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


cpp_fused_sum_23 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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


cpp_fused_sum_24 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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


cpp_fused_add_native_layer_norm_backward_sum_25 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr2[static_cast<long>(x0)];
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
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_add_native_layer_norm_backward_sum_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp2;
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


cpp_fused_add_native_layer_norm_backward_sum_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr2 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(8.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((512L*x1) + (512L*x1_inner) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>(x0) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                        tmp_acc2_vec = tmp_acc2_vec + tmp6;
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


cpp_fused_sum_35 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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


cpp_fused_sum_36 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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


cpp_fused_add_native_layer_norm_backward_sum_37 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr2[static_cast<long>(x0)];
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
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_add_native_layer_norm_backward_sum_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp2;
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


cpp_fused_add_native_layer_norm_backward_sum_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(8.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((512L*x1) + (512L*x1_inner) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>(x0) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                        tmp_acc2_vec = tmp_acc2_vec + tmp6;
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


cpp_fused_sum_47 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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


cpp_fused_sum_48 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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


cpp_fused_add_native_layer_norm_backward_sum_49 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr2[static_cast<long>(x0)];
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
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_add_native_layer_norm_backward_sum_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp2;
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


cpp_fused_add_native_layer_norm_backward_sum_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_55 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(8.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((512L*x1) + (512L*x1_inner) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>(x0) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                        tmp_acc2_vec = tmp_acc2_vec + tmp6;
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


cpp_fused_sum_59 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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


cpp_fused_sum_60 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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


cpp_fused_add_native_layer_norm_backward_sum_61 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr2[static_cast<long>(x0)];
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
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_add_native_layer_norm_backward_sum_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp2;
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


cpp_fused_add_native_layer_norm_backward_sum_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(8.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((512L*x1) + (512L*x1_inner) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>(x0) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                        tmp_acc2_vec = tmp_acc2_vec + tmp6;
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


cpp_fused_sum_71 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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


cpp_fused_sum_72 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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


cpp_fused_add_native_layer_norm_backward_sum_73 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr2[static_cast<long>(x0)];
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
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_add_native_layer_norm_backward_sum_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp2;
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


cpp_fused_add_native_layer_norm_backward_sum_76 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_79 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(8.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((512L*x1) + (512L*x1_inner) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>(x0) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_82 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                        tmp_acc2_vec = tmp_acc2_vec + tmp6;
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


cpp_fused_sum_83 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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


cpp_fused_sum_84 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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


cpp_fused_add_native_layer_norm_backward_sum_85 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr2[static_cast<long>(x0)];
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
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_add_native_layer_norm_backward_sum_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp2;
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


cpp_fused_add_native_layer_norm_backward_sum_88 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_91 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(8.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((512L*x1) + (512L*x1_inner) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>(x0) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_94 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                        tmp_acc2_vec = tmp_acc2_vec + tmp6;
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


cpp_fused_sum_95 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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


cpp_fused_sum_96 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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


cpp_fused_add_native_layer_norm_backward_sum_97 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr2[static_cast<long>(x0)];
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
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_add_native_layer_norm_backward_99 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_102 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(8.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((512L*x1) + (512L*x1_inner) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>(x0) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
    auto in_ptr1 = in_out_ptr0;
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                        tmp_acc2_vec = tmp_acc2_vec + tmp6;
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr3[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr4[static_cast<long>(x0)];
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
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_add_native_layer_norm_backward_107 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_110 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(8.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((512L*x1) + (512L*x1_inner) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>(x0) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
    auto in_ptr1 = in_out_ptr0;
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                        tmp_acc2_vec = tmp_acc2_vec + tmp6;
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr3[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr4[static_cast<long>(x0)];
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
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_add_native_layer_norm_backward_115 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_clone_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_118 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(8.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((512L*x1) + (512L*x1_inner) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>(x0) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_120 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       const float* in_ptr20,
                       const float* in_ptr21,
                       const float* in_ptr22,
                       const float* in_ptr23,
                       const float* in_ptr24,
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                        tmp_acc2_vec = tmp_acc2_vec + tmp6;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp14 = tmp12 + tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp22 = tmp20 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x0));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x0));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr19 + static_cast<long>(x0));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr20 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr21 + static_cast<long>(x0));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr22 + static_cast<long>(x0));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr23 + static_cast<long>(x0));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr24 + static_cast<long>(x0));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp14 = tmp12 + tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp22 = tmp20 + tmp21;
                    tmp22.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_122 = async_compile.cpp('''
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
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10)
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr5 + static_cast<long>(x0));
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr7 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr8 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp10;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr9[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr10[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr14[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr9[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr10[static_cast<long>(x0)];
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
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6291456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
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


cpp_fused_add_native_layer_norm_backward_sum_124 = async_compile.cpp('''
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
                       const float* in_ptr12,
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
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp2;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp14 = tmp12 + tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp22 = tmp20 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_125 = async_compile.cpp('''
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
                       const float* in_ptr10)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2359296L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                auto tmp10 = tmp8 + tmp9;
                auto tmp12 = tmp10 + tmp11;
                auto tmp14 = tmp12 + tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp18 = tmp16 + tmp17;
                auto tmp20 = tmp18 + tmp19;
                auto tmp22 = tmp20 + tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_sum_126 = async_compile.cpp('''
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
                       float* out_ptr3)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (3072L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp14 = tmp12 + tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp22 = tmp20 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_127 = async_compile.cpp('''
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
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       const float* in_ptr20,
                       const float* in_ptr21,
                       const float* in_ptr22,
                       const float* in_ptr23,
                       const float* in_ptr24,
                       const float* in_ptr25,
                       const float* in_ptr26,
                       const float* in_ptr27,
                       const float* in_ptr28,
                       const float* in_ptr29,
                       const float* in_ptr30,
                       const float* in_ptr31,
                       const float* in_ptr32)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2359296L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                auto tmp10 = tmp8 + tmp9;
                auto tmp12 = tmp10 + tmp11;
                auto tmp14 = tmp12 + tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp18 = tmp16 + tmp17;
                auto tmp20 = tmp18 + tmp19;
                auto tmp22 = tmp20 + tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x0));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x0));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x0));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr19 + static_cast<long>(x0));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr20 + static_cast<long>(x0));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr21 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp14 = tmp12 + tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp22 = tmp20 + tmp21;
                    tmp22.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr22 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr23 + static_cast<long>(x0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr24 + static_cast<long>(x0));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr25 + static_cast<long>(x0));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr26 + static_cast<long>(x0));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr27 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr28 + static_cast<long>(x0));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr29 + static_cast<long>(x0));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr30 + static_cast<long>(x0));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr31 + static_cast<long>(x0));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr32 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp14 = tmp12 + tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp22 = tmp20 + tmp21;
                    tmp22.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_128 = async_compile.cpp('''
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
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x0));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp14 = tmp12 + tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp22 = tmp20 + tmp21;
                    tmp22.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_129 = async_compile.cpp('''
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
                       const float* in_ptr10)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(589824L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                auto tmp10 = tmp8 + tmp9;
                auto tmp12 = tmp10 + tmp11;
                auto tmp14 = tmp12 + tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp18 = tmp16 + tmp17;
                auto tmp20 = tmp18 + tmp19;
                auto tmp22 = tmp20 + tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_130 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_131 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_sum_view_132 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp14 = tmp12 + tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp22 = tmp20 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_133 = async_compile.cpp('''
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
                       const float* in_ptr10)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(589824L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                auto tmp10 = tmp8 + tmp9;
                auto tmp12 = tmp10 + tmp11;
                auto tmp14 = tmp12 + tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp18 = tmp16 + tmp17;
                auto tmp20 = tmp18 + tmp19;
                auto tmp22 = tmp20 + tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_134 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_135 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp1 * tmp4;
                    auto tmp6 = tmp2 - tmp5;
                    auto tmp7 = static_cast<float>(8.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    tmp9.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_sum_136 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((512L*x1) + (512L*x1_inner) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>(x0) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp14 = tmp12 + tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp22 = tmp20 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_137 = async_compile.cpp('''
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
                       const float* in_ptr10)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(589824L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                auto tmp10 = tmp8 + tmp9;
                auto tmp12 = tmp10 + tmp11;
                auto tmp14 = tmp12 + tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp18 = tmp16 + tmp17;
                auto tmp20 = tmp18 + tmp19;
                auto tmp22 = tmp20 + tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_138 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_sum_view_139 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(512L))) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (393216L*(c10::div_floor_integer(x0, 512L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    auto tmp12 = tmp10 + tmp11;
                    auto tmp14 = tmp12 + tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    auto tmp22 = tmp20 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_140 = async_compile.cpp('''
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
                       const float* in_ptr10)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(589824L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                auto tmp10 = tmp8 + tmp9;
                auto tmp12 = tmp10 + tmp11;
                auto tmp14 = tmp12 + tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp18 = tmp16 + tmp17;
                auto tmp20 = tmp18 + tmp19;
                auto tmp22 = tmp20 + tmp21;
                tmp22.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_141 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_native_layer_norm_backward_sum_142 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const long* in_ptr5,
                       const long* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp11 = out_ptr2[static_cast<long>(x0)];
                    auto tmp17 = in_ptr5[static_cast<long>(x0)];
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
                    auto tmp18 = static_cast<int>(0);
                    auto tmp19 = tmp17 == tmp18;
                    auto tmp20 = static_cast<float>(0.0);
                    auto tmp21 = to_float_mask(tmp19);
                    auto tmp22 = at::vec::Vectorized<float>(tmp20);
                    auto tmp23 = decltype(tmp22)::blendv(tmp16, tmp22, tmp21);
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    tmp23.store(out_ptr4 + static_cast<long>(x1 + (128L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = static_cast<float>(0.0);
                    auto tmp1 = at::vec::Vectorized<float>(tmp0);
                    tmp1.store(out_ptr7 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr6[static_cast<long>(x0)];
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(65536L + x1 + (128L*x0)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(131072L + x1 + (128L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(196608L + x1 + (128L*x0)));
                        auto tmp1 = static_cast<int>(-1);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp10 = static_cast<float>(0.0);
                        auto tmp11 = to_float_mask(tmp2);
                        auto tmp12 = at::vec::Vectorized<float>(tmp10);
                        auto tmp13 = decltype(tmp12)::blendv(tmp9, tmp12, tmp11);
                        tmp13.store(out_ptr8 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_143 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (65536L*x0)));
                        auto tmp1 = static_cast<int>(-1);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = to_float_mask(tmp2);
                        auto tmp6 = at::vec::Vectorized<float>(tmp4);
                        auto tmp7 = decltype(tmp6)::blendv(tmp3, tmp6, tmp5);
                        tmp7.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_144 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840000L); x0+=static_cast<long>(8L))
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
    primals_4, primals_16, primals_22, primals_26, primals_32, expand, slice_2, mul_1, view, view_2, view_18, mul_3, view_20, addmm_5, tanh, view_22, mul_9, view_24, view_40, mul_11, view_42, addmm_11, tanh_1, view_44, mul_17, view_46, view_62, mul_19, view_64, addmm_17, tanh_2, view_66, mul_25, view_68, view_84, mul_27, view_86, addmm_23, tanh_3, view_88, mul_33, view_90, view_106, mul_35, view_108, addmm_29, tanh_4, view_110, mul_41, view_112, view_128, mul_43, view_130, addmm_35, tanh_5, view_132, mul_49, view_134, view_150, mul_51, view_152, addmm_41, tanh_6, view_154, mul_57, view_156, view_172, mul_59, view_174, addmm_47, tanh_7, view_176, mul_65, view_178, view_194, mul_67, view_196, addmm_53, tanh_8, view_198, mul_73, view_200, view_216, mul_75, view_218, addmm_59, tanh_9, view_220, mul_81, view_222, view_238, mul_83, view_240, addmm_65, tanh_10, view_242, mul_89, view_244, view_260, mul_91, view_262, addmm_71, tanh_11, view_264, mul_97, view_266, addmm_73, tanh_12, getitem_51, rsqrt_25, view_268, permute_135, permute_139, div_25, permute_143, permute_147, div_26, permute_151, permute_156, permute_157, alias_27, permute_158, permute_159, permute_164, permute_168, permute_172, div_28, div_29, permute_189, permute_190, alias_29, permute_191, permute_192, div_31, div_32, permute_222, permute_223, alias_31, permute_224, permute_225, div_34, div_35, permute_255, permute_256, alias_33, permute_257, permute_258, div_37, div_38, permute_288, permute_289, alias_35, permute_290, permute_291, div_40, div_41, permute_321, permute_322, alias_37, permute_323, permute_324, div_43, div_44, permute_354, permute_355, alias_39, permute_356, permute_357, div_46, div_47, permute_387, permute_388, alias_41, permute_389, permute_390, div_49, div_50, permute_420, permute_421, alias_43, permute_422, permute_423, div_52, div_53, permute_453, permute_454, alias_45, permute_455, permute_456, div_55, div_56, permute_486, permute_487, alias_47, permute_488, permute_489, div_58, div_59, permute_519, permute_520, alias_49, permute_521, permute_522, permute_539, div_61, tangents_1 = args
    args.clear()
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_32, (4, 512), (512, 1))
    assert_size_stride(expand, (4, 512), (0, 1))
    assert_size_stride(slice_2, (1, 512), (512, 1))
    assert_size_stride(mul_1, (4, 512, 128), (65536, 128, 1))
    assert_size_stride(view, (2048, 128), (128, 1))
    assert_size_stride(view_2, (2048, 768), (768, 1))
    assert_size_stride(view_18, (2048, 768), (768, 1))
    assert_size_stride(mul_3, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_20, (2048, 768), (768, 1))
    assert_size_stride(addmm_5, (2048, 3072), (3072, 1))
    assert_size_stride(tanh, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_22, (2048, 3072), (3072, 1))
    assert_size_stride(mul_9, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_24, (2048, 768), (768, 1))
    assert_size_stride(view_40, (2048, 768), (768, 1))
    assert_size_stride(mul_11, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_42, (2048, 768), (768, 1))
    assert_size_stride(addmm_11, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_1, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_44, (2048, 3072), (3072, 1))
    assert_size_stride(mul_17, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_46, (2048, 768), (768, 1))
    assert_size_stride(view_62, (2048, 768), (768, 1))
    assert_size_stride(mul_19, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_64, (2048, 768), (768, 1))
    assert_size_stride(addmm_17, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_2, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_66, (2048, 3072), (3072, 1))
    assert_size_stride(mul_25, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_68, (2048, 768), (768, 1))
    assert_size_stride(view_84, (2048, 768), (768, 1))
    assert_size_stride(mul_27, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_86, (2048, 768), (768, 1))
    assert_size_stride(addmm_23, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_3, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_88, (2048, 3072), (3072, 1))
    assert_size_stride(mul_33, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_90, (2048, 768), (768, 1))
    assert_size_stride(view_106, (2048, 768), (768, 1))
    assert_size_stride(mul_35, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_108, (2048, 768), (768, 1))
    assert_size_stride(addmm_29, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_4, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_110, (2048, 3072), (3072, 1))
    assert_size_stride(mul_41, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_112, (2048, 768), (768, 1))
    assert_size_stride(view_128, (2048, 768), (768, 1))
    assert_size_stride(mul_43, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_130, (2048, 768), (768, 1))
    assert_size_stride(addmm_35, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_5, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_132, (2048, 3072), (3072, 1))
    assert_size_stride(mul_49, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_134, (2048, 768), (768, 1))
    assert_size_stride(view_150, (2048, 768), (768, 1))
    assert_size_stride(mul_51, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_152, (2048, 768), (768, 1))
    assert_size_stride(addmm_41, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_6, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_154, (2048, 3072), (3072, 1))
    assert_size_stride(mul_57, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_156, (2048, 768), (768, 1))
    assert_size_stride(view_172, (2048, 768), (768, 1))
    assert_size_stride(mul_59, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_174, (2048, 768), (768, 1))
    assert_size_stride(addmm_47, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_7, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_176, (2048, 3072), (3072, 1))
    assert_size_stride(mul_65, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_178, (2048, 768), (768, 1))
    assert_size_stride(view_194, (2048, 768), (768, 1))
    assert_size_stride(mul_67, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_196, (2048, 768), (768, 1))
    assert_size_stride(addmm_53, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_8, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_198, (2048, 3072), (3072, 1))
    assert_size_stride(mul_73, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_200, (2048, 768), (768, 1))
    assert_size_stride(view_216, (2048, 768), (768, 1))
    assert_size_stride(mul_75, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_218, (2048, 768), (768, 1))
    assert_size_stride(addmm_59, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_9, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_220, (2048, 3072), (3072, 1))
    assert_size_stride(mul_81, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_222, (2048, 768), (768, 1))
    assert_size_stride(view_238, (2048, 768), (768, 1))
    assert_size_stride(mul_83, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_240, (2048, 768), (768, 1))
    assert_size_stride(addmm_65, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_10, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_242, (2048, 3072), (3072, 1))
    assert_size_stride(mul_89, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_244, (2048, 768), (768, 1))
    assert_size_stride(view_260, (2048, 768), (768, 1))
    assert_size_stride(mul_91, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_262, (2048, 768), (768, 1))
    assert_size_stride(addmm_71, (2048, 3072), (3072, 1))
    assert_size_stride(tanh_11, (4, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(view_264, (2048, 3072), (3072, 1))
    assert_size_stride(mul_97, (4, 512, 768), (393216, 768, 1))
    assert_size_stride(view_266, (2048, 768), (768, 1))
    assert_size_stride(addmm_73, (2048, 128), (128, 1))
    assert_size_stride(tanh_12, (4, 512, 128), (65536, 128, 1))
    assert_size_stride(getitem_51, (4, 512, 1), (512, 1, 1))
    assert_size_stride(rsqrt_25, (4, 512, 1), (512, 1, 1))
    assert_size_stride(view_268, (2048, 128), (128, 1))
    assert_size_stride(permute_135, (30000, 128), (128, 1))
    assert_size_stride(permute_139, (128, 768), (768, 1))
    assert_size_stride(div_25, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_143, (768, 3072), (3072, 1))
    assert_size_stride(permute_147, (3072, 768), (768, 1))
    assert_size_stride(div_26, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_151, (768, 768), (768, 1))
    assert_size_stride(permute_156, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_157, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_27, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_158, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_159, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_164, (768, 768), (768, 1))
    assert_size_stride(permute_168, (768, 768), (768, 1))
    assert_size_stride(permute_172, (768, 768), (768, 1))
    assert_size_stride(div_28, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_29, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_189, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_190, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_29, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_191, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_192, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_31, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_32, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_222, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_223, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_31, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_224, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_225, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_34, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_35, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_255, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_256, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_33, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_257, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_258, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_37, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_38, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_288, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_289, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_35, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_290, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_291, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_40, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_41, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_321, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_322, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_37, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_323, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_324, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_43, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_44, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_354, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_355, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_39, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_356, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_357, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_46, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_47, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_387, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_388, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_41, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_389, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_390, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_49, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_50, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_420, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_421, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_43, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_422, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_423, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_52, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_53, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_453, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_454, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_45, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_455, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_456, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_55, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_56, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_486, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_487, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_47, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_488, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_489, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(div_58, (4, 512, 1), (512, 1, 1))
    assert_size_stride(div_59, (4, 512, 1), (512, 1, 1))
    assert_size_stride(permute_519, (48, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_520, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_49, (4, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_521, (48, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_522, (48, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_539, (768, 128), (128, 1))
    assert_size_stride(div_61, (4, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (4, 512, 30000), (15360000, 30000, 1))
    buf0 = empty((2048, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (2048, 30000), (30000, 1), 0), permute_135, out=buf0)
    del permute_135
    buf1 = empty((30000, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (30000, 2048), (1, 30000), 0), view_268, out=buf1)
    del view_268
    buf2 = empty((1, 30000), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((4, 512, 1), (512, 1, 2048), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((4, 512, 1), (512, 1, 2048), device='cpu', dtype=torch.float32)
    buf5 = empty((4, 512, 128), device='cpu', dtype=torch.float32)
    buf6 = empty((128, ), device='cpu', dtype=torch.float32)
    buf7 = empty((128, ), device='cpu', dtype=torch.float32)
    buf8 = buf5; del buf5  # reuse
    cpp_fused_add_mul_native_layer_norm_native_layer_norm_backward_pow_sum_tanh_backward_0(c_void_p(buf8.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(addmm_73.data_ptr()), c_void_p(tanh_12.data_ptr()), c_void_p(getitem_51.data_ptr()), c_void_p(rsqrt_25.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del addmm_73
    del getitem_51
    del primals_26
    del rsqrt_25
    del tangents_1
    del tanh_12
    buf9 = empty((2048, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf8, (2048, 128), (128, 1), 0), permute_139, out=buf9)
    del permute_139
    buf10 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf8, (128, 2048), (1, 128), 0), view_266, out=buf10)
    del view_266
    buf11 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf12 = buf4; del buf4  # reuse
    buf13 = buf3; del buf3  # reuse
    buf14 = empty((4, 512, 768), device='cpu', dtype=torch.float32)
    buf15 = empty((768, ), device='cpu', dtype=torch.float32)
    buf16 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_backward_sum_1(c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_97.data_ptr()), c_void_p(div_25.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()))
    del div_25
    del mul_97
    buf17 = empty((2048, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf14, (2048, 768), (768, 1), 0), permute_143, out=buf17)
    buf18 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf14, (768, 2048), (1, 768), 0), view_264, out=buf18)
    del view_264
    buf20 = reinterpret_tensor(buf17, (4, 512, 3072), (1572864, 3072, 1), 0); del buf17  # reuse
    cpp_fused_add_mul_pow_tanh_backward_2(c_void_p(buf20.data_ptr()), c_void_p(addmm_71.data_ptr()), c_void_p(tanh_11.data_ptr()))
    del addmm_71
    del tanh_11
    buf21 = buf9; del buf9  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf20, (2048, 3072), (3072, 1), 0), permute_147, out=buf21)
    buf19 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf27 = empty((768, ), device='cpu', dtype=torch.float32)
    buf28 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_3(c_void_p(buf14.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(mul_91.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()))
    buf22 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf20, (3072, 2048), (1, 3072), 0), view_262, out=buf22)
    del view_262
    buf23 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf24 = buf13; del buf13  # reuse
    buf25 = buf12; del buf12  # reuse
    buf26 = buf14; del buf14  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_4(c_void_p(buf26.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_91.data_ptr()), c_void_p(div_26.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()))
    del div_26
    del mul_91
    buf29 = buf21; del buf21  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf26, (2048, 768), (768, 1), 0), permute_151, out=buf29)
    buf30 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf26, (768, 2048), (1, 768), 0), view_260, out=buf30)
    del view_260
    buf32 = empty((4, 12, 512, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_5(c_void_p(buf29.data_ptr()), c_void_p(buf32.data_ptr()))
    buf33 = reinterpret_tensor(buf29, (48, 512, 64), (32768, 64, 1), 0); del buf29  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_156, reinterpret_tensor(buf32, (48, 512, 64), (32768, 64, 1), 0), out=buf33)
    del permute_156
    buf39 = empty((2048, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_6(c_void_p(buf33.data_ptr()), c_void_p(buf39.data_ptr()))
    buf40 = reinterpret_tensor(buf33, (2048, 768), (768, 1), 0); del buf33  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf39, permute_164, out=buf40)
    buf34 = empty((48, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf32, (48, 512, 64), (32768, 64, 1), 0), permute_157, out=buf34)
    del permute_157
    buf35 = empty_strided((4, 12, 512, 1), (6144, 512, 1, 24576), device='cpu', dtype=torch.float32)
    buf36 = reinterpret_tensor(buf34, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf34  # reuse
    cpp_fused__softmax_backward_data_div_7(c_void_p(buf36.data_ptr()), c_void_p(alias_27.data_ptr()), c_void_p(buf35.data_ptr()))
    del alias_27
    buf37 = reinterpret_tensor(buf32, (48, 64, 512), (32768, 512, 1), 0); del buf32  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_158, reinterpret_tensor(buf36, (48, 512, 512), (262144, 512, 1), 0), out=buf37)
    del permute_158
    buf43 = empty((2048, 768), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_8(c_void_p(buf37.data_ptr()), c_void_p(buf43.data_ptr()))
    buf44 = reinterpret_tensor(buf37, (2048, 768), (768, 1), 0); del buf37  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf43, permute_168, out=buf44)
    buf38 = empty((48, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf36, (48, 512, 512), (262144, 512, 1), 0), permute_159, out=buf38)
    del permute_159
    buf47 = empty((2048, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_9(c_void_p(buf38.data_ptr()), c_void_p(buf47.data_ptr()))
    buf48 = reinterpret_tensor(buf38, (2048, 768), (768, 1), 0); del buf38  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf47, permute_172, out=buf48)
    buf31 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf55 = empty((768, ), device='cpu', dtype=torch.float32)
    buf56 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_10(c_void_p(buf26.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(mul_89.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()))
    buf41 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf39, (768, 2048), (1, 768), 0), view_244, out=buf41)
    buf42 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_11(c_void_p(buf39.data_ptr()), c_void_p(buf42.data_ptr()))
    buf45 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf43, (768, 2048), (1, 768), 0), view_244, out=buf45)
    buf46 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_12(c_void_p(buf43.data_ptr()), c_void_p(buf46.data_ptr()))
    buf49 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf47, (768, 2048), (1, 768), 0), view_244, out=buf49)
    del view_244
    buf50 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf51 = buf26; del buf26  # reuse
    buf52 = buf25; del buf25  # reuse
    buf53 = buf24; del buf24  # reuse
    buf54 = buf51; del buf51  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_13(c_void_p(buf54.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_89.data_ptr()), c_void_p(div_28.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()))
    del div_28
    del mul_89
    buf57 = reinterpret_tensor(buf20, (2048, 3072), (3072, 1), 0); del buf20  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf54, (2048, 768), (768, 1), 0), permute_143, out=buf57)
    buf58 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf54, (768, 2048), (1, 768), 0), view_242, out=buf58)
    del view_242
    buf60 = reinterpret_tensor(buf57, (4, 512, 3072), (1572864, 3072, 1), 0); del buf57  # reuse
    cpp_fused_add_mul_pow_tanh_backward_14(c_void_p(buf60.data_ptr()), c_void_p(addmm_65.data_ptr()), c_void_p(tanh_10.data_ptr()))
    del addmm_65
    del tanh_10
    buf61 = buf48; del buf48  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf60, (2048, 3072), (3072, 1), 0), permute_147, out=buf61)
    buf59 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf67 = empty((768, ), device='cpu', dtype=torch.float32)
    buf68 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_15(c_void_p(buf54.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(mul_83.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()))
    buf62 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf60, (3072, 2048), (1, 3072), 0), view_240, out=buf62)
    del view_240
    buf63 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf64 = buf53; del buf53  # reuse
    buf65 = buf52; del buf52  # reuse
    buf66 = buf54; del buf54  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_16(c_void_p(buf66.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_83.data_ptr()), c_void_p(div_29.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()))
    del div_29
    del mul_83
    buf69 = buf61; del buf61  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf66, (2048, 768), (768, 1), 0), permute_151, out=buf69)
    buf70 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf66, (768, 2048), (1, 768), 0), view_238, out=buf70)
    del view_238
    buf72 = reinterpret_tensor(buf47, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf47  # reuse
    cpp_fused_clone_17(c_void_p(buf69.data_ptr()), c_void_p(buf72.data_ptr()))
    buf73 = reinterpret_tensor(buf69, (48, 512, 64), (32768, 64, 1), 0); del buf69  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_189, reinterpret_tensor(buf72, (48, 512, 64), (32768, 64, 1), 0), out=buf73)
    del permute_189
    buf79 = buf44; del buf44  # reuse
    cpp_fused_view_18(c_void_p(buf73.data_ptr()), c_void_p(buf79.data_ptr()))
    buf80 = reinterpret_tensor(buf73, (2048, 768), (768, 1), 0); del buf73  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf79, permute_164, out=buf80)
    buf74 = reinterpret_tensor(buf36, (48, 512, 512), (262144, 512, 1), 0); del buf36  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf72, (48, 512, 64), (32768, 64, 1), 0), permute_190, out=buf74)
    del permute_190
    buf75 = buf35; del buf35  # reuse
    buf76 = reinterpret_tensor(buf74, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf74  # reuse
    cpp_fused__softmax_backward_data_div_19(c_void_p(buf76.data_ptr()), c_void_p(alias_29.data_ptr()), c_void_p(buf75.data_ptr()))
    del alias_29
    buf77 = reinterpret_tensor(buf72, (48, 64, 512), (32768, 512, 1), 0); del buf72  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_191, reinterpret_tensor(buf76, (48, 512, 512), (262144, 512, 1), 0), out=buf77)
    del permute_191
    buf83 = buf40; del buf40  # reuse
    cpp_fused__unsafe_view_clone_20(c_void_p(buf77.data_ptr()), c_void_p(buf83.data_ptr()))
    buf84 = reinterpret_tensor(buf77, (2048, 768), (768, 1), 0); del buf77  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf83, permute_168, out=buf84)
    buf78 = reinterpret_tensor(buf43, (48, 512, 64), (32768, 64, 1), 0); del buf43  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf76, (48, 512, 512), (262144, 512, 1), 0), permute_192, out=buf78)
    del permute_192
    buf87 = buf39; del buf39  # reuse
    cpp_fused_view_21(c_void_p(buf78.data_ptr()), c_void_p(buf87.data_ptr()))
    buf88 = reinterpret_tensor(buf78, (2048, 768), (768, 1), 0); del buf78  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf87, permute_172, out=buf88)
    buf71 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf95 = empty((768, ), device='cpu', dtype=torch.float32)
    buf96 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_22(c_void_p(buf66.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(mul_81.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()))
    buf81 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf79, (768, 2048), (1, 768), 0), view_222, out=buf81)
    buf82 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_23(c_void_p(buf79.data_ptr()), c_void_p(buf82.data_ptr()))
    buf85 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf83, (768, 2048), (1, 768), 0), view_222, out=buf85)
    buf86 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_24(c_void_p(buf83.data_ptr()), c_void_p(buf86.data_ptr()))
    buf89 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf87, (768, 2048), (1, 768), 0), view_222, out=buf89)
    del view_222
    buf90 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf91 = buf66; del buf66  # reuse
    buf92 = buf65; del buf65  # reuse
    buf93 = buf64; del buf64  # reuse
    buf94 = buf91; del buf91  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_25(c_void_p(buf94.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_81.data_ptr()), c_void_p(div_31.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()))
    del div_31
    del mul_81
    buf97 = reinterpret_tensor(buf60, (2048, 3072), (3072, 1), 0); del buf60  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf94, (2048, 768), (768, 1), 0), permute_143, out=buf97)
    buf98 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf94, (768, 2048), (1, 768), 0), view_220, out=buf98)
    del view_220
    buf100 = reinterpret_tensor(buf97, (4, 512, 3072), (1572864, 3072, 1), 0); del buf97  # reuse
    cpp_fused_add_mul_pow_tanh_backward_26(c_void_p(buf100.data_ptr()), c_void_p(addmm_59.data_ptr()), c_void_p(tanh_9.data_ptr()))
    del addmm_59
    del tanh_9
    buf101 = buf88; del buf88  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf100, (2048, 3072), (3072, 1), 0), permute_147, out=buf101)
    buf99 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf107 = empty((768, ), device='cpu', dtype=torch.float32)
    buf108 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_27(c_void_p(buf94.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(mul_75.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()))
    buf102 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf100, (3072, 2048), (1, 3072), 0), view_218, out=buf102)
    del view_218
    buf103 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf104 = buf93; del buf93  # reuse
    buf105 = buf92; del buf92  # reuse
    buf106 = reinterpret_tensor(buf101, (4, 512, 768), (393216, 768, 1), 0); del buf101  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_28(c_void_p(buf106.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_75.data_ptr()), c_void_p(div_32.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()))
    del div_32
    del mul_75
    buf109 = reinterpret_tensor(buf94, (2048, 768), (768, 1), 0); del buf94  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf106, (2048, 768), (768, 1), 0), permute_151, out=buf109)
    buf110 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf106, (768, 2048), (1, 768), 0), view_216, out=buf110)
    del view_216
    buf112 = reinterpret_tensor(buf87, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf87  # reuse
    cpp_fused_clone_29(c_void_p(buf109.data_ptr()), c_void_p(buf112.data_ptr()))
    buf113 = reinterpret_tensor(buf109, (48, 512, 64), (32768, 64, 1), 0); del buf109  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_222, reinterpret_tensor(buf112, (48, 512, 64), (32768, 64, 1), 0), out=buf113)
    del permute_222
    buf119 = buf84; del buf84  # reuse
    cpp_fused_view_30(c_void_p(buf113.data_ptr()), c_void_p(buf119.data_ptr()))
    buf120 = reinterpret_tensor(buf113, (2048, 768), (768, 1), 0); del buf113  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf119, permute_164, out=buf120)
    buf114 = reinterpret_tensor(buf76, (48, 512, 512), (262144, 512, 1), 0); del buf76  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf112, (48, 512, 64), (32768, 64, 1), 0), permute_223, out=buf114)
    del permute_223
    buf115 = buf75; del buf75  # reuse
    buf116 = reinterpret_tensor(buf114, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf114  # reuse
    cpp_fused__softmax_backward_data_div_31(c_void_p(buf116.data_ptr()), c_void_p(alias_31.data_ptr()), c_void_p(buf115.data_ptr()))
    del alias_31
    buf117 = reinterpret_tensor(buf112, (48, 64, 512), (32768, 512, 1), 0); del buf112  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_224, reinterpret_tensor(buf116, (48, 512, 512), (262144, 512, 1), 0), out=buf117)
    del permute_224
    buf123 = buf80; del buf80  # reuse
    cpp_fused__unsafe_view_clone_32(c_void_p(buf117.data_ptr()), c_void_p(buf123.data_ptr()))
    buf124 = reinterpret_tensor(buf117, (2048, 768), (768, 1), 0); del buf117  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf123, permute_168, out=buf124)
    buf118 = reinterpret_tensor(buf83, (48, 512, 64), (32768, 64, 1), 0); del buf83  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf116, (48, 512, 512), (262144, 512, 1), 0), permute_225, out=buf118)
    del permute_225
    buf127 = buf79; del buf79  # reuse
    cpp_fused_view_33(c_void_p(buf118.data_ptr()), c_void_p(buf127.data_ptr()))
    buf128 = reinterpret_tensor(buf118, (2048, 768), (768, 1), 0); del buf118  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf127, permute_172, out=buf128)
    buf111 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf135 = empty((768, ), device='cpu', dtype=torch.float32)
    buf136 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_34(c_void_p(buf106.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(mul_73.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()))
    buf121 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf119, (768, 2048), (1, 768), 0), view_200, out=buf121)
    buf122 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_35(c_void_p(buf119.data_ptr()), c_void_p(buf122.data_ptr()))
    buf125 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf123, (768, 2048), (1, 768), 0), view_200, out=buf125)
    buf126 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_36(c_void_p(buf123.data_ptr()), c_void_p(buf126.data_ptr()))
    buf129 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf127, (768, 2048), (1, 768), 0), view_200, out=buf129)
    del view_200
    buf130 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf131 = buf106; del buf106  # reuse
    buf132 = buf105; del buf105  # reuse
    buf133 = buf104; del buf104  # reuse
    buf134 = buf131; del buf131  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_37(c_void_p(buf134.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_73.data_ptr()), c_void_p(div_34.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()))
    del div_34
    del mul_73
    buf137 = reinterpret_tensor(buf100, (2048, 3072), (3072, 1), 0); del buf100  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf134, (2048, 768), (768, 1), 0), permute_143, out=buf137)
    buf138 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf134, (768, 2048), (1, 768), 0), view_198, out=buf138)
    del view_198
    buf140 = reinterpret_tensor(buf137, (4, 512, 3072), (1572864, 3072, 1), 0); del buf137  # reuse
    cpp_fused_add_mul_pow_tanh_backward_38(c_void_p(buf140.data_ptr()), c_void_p(addmm_53.data_ptr()), c_void_p(tanh_8.data_ptr()))
    del addmm_53
    del tanh_8
    buf141 = buf128; del buf128  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf140, (2048, 3072), (3072, 1), 0), permute_147, out=buf141)
    buf139 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf147 = empty((768, ), device='cpu', dtype=torch.float32)
    buf148 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_39(c_void_p(buf134.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(mul_67.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()))
    buf142 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf140, (3072, 2048), (1, 3072), 0), view_196, out=buf142)
    del view_196
    buf143 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf144 = buf133; del buf133  # reuse
    buf145 = buf132; del buf132  # reuse
    buf146 = buf134; del buf134  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_40(c_void_p(buf146.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_67.data_ptr()), c_void_p(div_35.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    del div_35
    del mul_67
    buf149 = buf141; del buf141  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf146, (2048, 768), (768, 1), 0), permute_151, out=buf149)
    buf150 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf146, (768, 2048), (1, 768), 0), view_194, out=buf150)
    del view_194
    buf152 = reinterpret_tensor(buf127, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf127  # reuse
    cpp_fused_clone_41(c_void_p(buf149.data_ptr()), c_void_p(buf152.data_ptr()))
    buf153 = reinterpret_tensor(buf149, (48, 512, 64), (32768, 64, 1), 0); del buf149  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_255, reinterpret_tensor(buf152, (48, 512, 64), (32768, 64, 1), 0), out=buf153)
    del permute_255
    buf159 = buf124; del buf124  # reuse
    cpp_fused_view_42(c_void_p(buf153.data_ptr()), c_void_p(buf159.data_ptr()))
    buf160 = reinterpret_tensor(buf153, (2048, 768), (768, 1), 0); del buf153  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf159, permute_164, out=buf160)
    buf154 = reinterpret_tensor(buf116, (48, 512, 512), (262144, 512, 1), 0); del buf116  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf152, (48, 512, 64), (32768, 64, 1), 0), permute_256, out=buf154)
    del permute_256
    buf155 = buf115; del buf115  # reuse
    buf156 = reinterpret_tensor(buf154, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf154  # reuse
    cpp_fused__softmax_backward_data_div_43(c_void_p(buf156.data_ptr()), c_void_p(alias_33.data_ptr()), c_void_p(buf155.data_ptr()))
    del alias_33
    buf157 = reinterpret_tensor(buf152, (48, 64, 512), (32768, 512, 1), 0); del buf152  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_257, reinterpret_tensor(buf156, (48, 512, 512), (262144, 512, 1), 0), out=buf157)
    del permute_257
    buf163 = buf120; del buf120  # reuse
    cpp_fused__unsafe_view_clone_44(c_void_p(buf157.data_ptr()), c_void_p(buf163.data_ptr()))
    buf164 = reinterpret_tensor(buf157, (2048, 768), (768, 1), 0); del buf157  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf163, permute_168, out=buf164)
    buf158 = reinterpret_tensor(buf123, (48, 512, 64), (32768, 64, 1), 0); del buf123  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf156, (48, 512, 512), (262144, 512, 1), 0), permute_258, out=buf158)
    del permute_258
    buf167 = buf119; del buf119  # reuse
    cpp_fused_view_45(c_void_p(buf158.data_ptr()), c_void_p(buf167.data_ptr()))
    buf168 = reinterpret_tensor(buf158, (2048, 768), (768, 1), 0); del buf158  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf167, permute_172, out=buf168)
    buf151 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf175 = empty((768, ), device='cpu', dtype=torch.float32)
    buf176 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_46(c_void_p(buf146.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(mul_65.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()))
    buf161 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf159, (768, 2048), (1, 768), 0), view_178, out=buf161)
    buf162 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_47(c_void_p(buf159.data_ptr()), c_void_p(buf162.data_ptr()))
    buf165 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf163, (768, 2048), (1, 768), 0), view_178, out=buf165)
    buf166 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_48(c_void_p(buf163.data_ptr()), c_void_p(buf166.data_ptr()))
    buf169 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf167, (768, 2048), (1, 768), 0), view_178, out=buf169)
    del view_178
    buf170 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf171 = buf146; del buf146  # reuse
    buf172 = buf145; del buf145  # reuse
    buf173 = buf144; del buf144  # reuse
    buf174 = buf171; del buf171  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_49(c_void_p(buf174.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_65.data_ptr()), c_void_p(div_37.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()))
    del div_37
    del mul_65
    buf177 = reinterpret_tensor(buf140, (2048, 3072), (3072, 1), 0); del buf140  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (2048, 768), (768, 1), 0), permute_143, out=buf177)
    buf178 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (768, 2048), (1, 768), 0), view_176, out=buf178)
    del view_176
    buf180 = reinterpret_tensor(buf177, (4, 512, 3072), (1572864, 3072, 1), 0); del buf177  # reuse
    cpp_fused_add_mul_pow_tanh_backward_50(c_void_p(buf180.data_ptr()), c_void_p(addmm_47.data_ptr()), c_void_p(tanh_7.data_ptr()))
    del addmm_47
    del tanh_7
    buf181 = buf168; del buf168  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf180, (2048, 3072), (3072, 1), 0), permute_147, out=buf181)
    buf179 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf187 = empty((768, ), device='cpu', dtype=torch.float32)
    buf188 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_51(c_void_p(buf174.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(mul_59.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()))
    buf182 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf180, (3072, 2048), (1, 3072), 0), view_174, out=buf182)
    del view_174
    buf183 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf184 = buf173; del buf173  # reuse
    buf185 = buf172; del buf172  # reuse
    buf186 = buf174; del buf174  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_52(c_void_p(buf186.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_59.data_ptr()), c_void_p(div_38.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()))
    del div_38
    del mul_59
    buf189 = buf181; del buf181  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf186, (2048, 768), (768, 1), 0), permute_151, out=buf189)
    buf190 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf186, (768, 2048), (1, 768), 0), view_172, out=buf190)
    del view_172
    buf192 = reinterpret_tensor(buf167, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf167  # reuse
    cpp_fused_clone_53(c_void_p(buf189.data_ptr()), c_void_p(buf192.data_ptr()))
    buf193 = reinterpret_tensor(buf189, (48, 512, 64), (32768, 64, 1), 0); del buf189  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_288, reinterpret_tensor(buf192, (48, 512, 64), (32768, 64, 1), 0), out=buf193)
    del permute_288
    buf199 = buf164; del buf164  # reuse
    cpp_fused_view_54(c_void_p(buf193.data_ptr()), c_void_p(buf199.data_ptr()))
    buf200 = reinterpret_tensor(buf193, (2048, 768), (768, 1), 0); del buf193  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf199, permute_164, out=buf200)
    buf194 = reinterpret_tensor(buf156, (48, 512, 512), (262144, 512, 1), 0); del buf156  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf192, (48, 512, 64), (32768, 64, 1), 0), permute_289, out=buf194)
    del permute_289
    buf195 = buf155; del buf155  # reuse
    buf196 = reinterpret_tensor(buf194, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf194  # reuse
    cpp_fused__softmax_backward_data_div_55(c_void_p(buf196.data_ptr()), c_void_p(alias_35.data_ptr()), c_void_p(buf195.data_ptr()))
    del alias_35
    buf197 = reinterpret_tensor(buf192, (48, 64, 512), (32768, 512, 1), 0); del buf192  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_290, reinterpret_tensor(buf196, (48, 512, 512), (262144, 512, 1), 0), out=buf197)
    del permute_290
    buf203 = buf160; del buf160  # reuse
    cpp_fused__unsafe_view_clone_56(c_void_p(buf197.data_ptr()), c_void_p(buf203.data_ptr()))
    buf204 = reinterpret_tensor(buf197, (2048, 768), (768, 1), 0); del buf197  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf203, permute_168, out=buf204)
    buf198 = reinterpret_tensor(buf163, (48, 512, 64), (32768, 64, 1), 0); del buf163  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf196, (48, 512, 512), (262144, 512, 1), 0), permute_291, out=buf198)
    del permute_291
    buf207 = buf159; del buf159  # reuse
    cpp_fused_view_57(c_void_p(buf198.data_ptr()), c_void_p(buf207.data_ptr()))
    buf208 = reinterpret_tensor(buf198, (2048, 768), (768, 1), 0); del buf198  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf207, permute_172, out=buf208)
    buf191 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf215 = empty((768, ), device='cpu', dtype=torch.float32)
    buf216 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_58(c_void_p(buf186.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(mul_57.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()))
    buf201 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf199, (768, 2048), (1, 768), 0), view_156, out=buf201)
    buf202 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_59(c_void_p(buf199.data_ptr()), c_void_p(buf202.data_ptr()))
    buf205 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf203, (768, 2048), (1, 768), 0), view_156, out=buf205)
    buf206 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_60(c_void_p(buf203.data_ptr()), c_void_p(buf206.data_ptr()))
    buf209 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf207, (768, 2048), (1, 768), 0), view_156, out=buf209)
    del view_156
    buf210 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf211 = buf186; del buf186  # reuse
    buf212 = buf185; del buf185  # reuse
    buf213 = buf184; del buf184  # reuse
    buf214 = buf211; del buf211  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_61(c_void_p(buf214.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_57.data_ptr()), c_void_p(div_40.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()))
    del div_40
    del mul_57
    buf217 = reinterpret_tensor(buf180, (2048, 3072), (3072, 1), 0); del buf180  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf214, (2048, 768), (768, 1), 0), permute_143, out=buf217)
    buf218 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf214, (768, 2048), (1, 768), 0), view_154, out=buf218)
    del view_154
    buf220 = reinterpret_tensor(buf217, (4, 512, 3072), (1572864, 3072, 1), 0); del buf217  # reuse
    cpp_fused_add_mul_pow_tanh_backward_62(c_void_p(buf220.data_ptr()), c_void_p(addmm_41.data_ptr()), c_void_p(tanh_6.data_ptr()))
    del addmm_41
    del tanh_6
    buf221 = buf208; del buf208  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf220, (2048, 3072), (3072, 1), 0), permute_147, out=buf221)
    buf219 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf227 = empty((768, ), device='cpu', dtype=torch.float32)
    buf228 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_63(c_void_p(buf214.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(mul_51.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()))
    buf222 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf220, (3072, 2048), (1, 3072), 0), view_152, out=buf222)
    del view_152
    buf223 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf224 = buf213; del buf213  # reuse
    buf225 = buf212; del buf212  # reuse
    buf226 = buf214; del buf214  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_64(c_void_p(buf226.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_51.data_ptr()), c_void_p(div_41.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()))
    del div_41
    del mul_51
    buf229 = buf221; del buf221  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf226, (2048, 768), (768, 1), 0), permute_151, out=buf229)
    buf230 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf226, (768, 2048), (1, 768), 0), view_150, out=buf230)
    del view_150
    buf232 = reinterpret_tensor(buf207, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf207  # reuse
    cpp_fused_clone_65(c_void_p(buf229.data_ptr()), c_void_p(buf232.data_ptr()))
    buf233 = reinterpret_tensor(buf229, (48, 512, 64), (32768, 64, 1), 0); del buf229  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_321, reinterpret_tensor(buf232, (48, 512, 64), (32768, 64, 1), 0), out=buf233)
    del permute_321
    buf239 = buf204; del buf204  # reuse
    cpp_fused_view_66(c_void_p(buf233.data_ptr()), c_void_p(buf239.data_ptr()))
    buf240 = reinterpret_tensor(buf233, (2048, 768), (768, 1), 0); del buf233  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf239, permute_164, out=buf240)
    buf234 = reinterpret_tensor(buf196, (48, 512, 512), (262144, 512, 1), 0); del buf196  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf232, (48, 512, 64), (32768, 64, 1), 0), permute_322, out=buf234)
    del permute_322
    buf235 = buf195; del buf195  # reuse
    buf236 = reinterpret_tensor(buf234, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf234  # reuse
    cpp_fused__softmax_backward_data_div_67(c_void_p(buf236.data_ptr()), c_void_p(alias_37.data_ptr()), c_void_p(buf235.data_ptr()))
    del alias_37
    buf237 = reinterpret_tensor(buf232, (48, 64, 512), (32768, 512, 1), 0); del buf232  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_323, reinterpret_tensor(buf236, (48, 512, 512), (262144, 512, 1), 0), out=buf237)
    del permute_323
    buf243 = buf200; del buf200  # reuse
    cpp_fused__unsafe_view_clone_68(c_void_p(buf237.data_ptr()), c_void_p(buf243.data_ptr()))
    buf244 = reinterpret_tensor(buf237, (2048, 768), (768, 1), 0); del buf237  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf243, permute_168, out=buf244)
    buf238 = reinterpret_tensor(buf203, (48, 512, 64), (32768, 64, 1), 0); del buf203  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf236, (48, 512, 512), (262144, 512, 1), 0), permute_324, out=buf238)
    del permute_324
    buf247 = buf199; del buf199  # reuse
    cpp_fused_view_69(c_void_p(buf238.data_ptr()), c_void_p(buf247.data_ptr()))
    buf248 = reinterpret_tensor(buf238, (2048, 768), (768, 1), 0); del buf238  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf247, permute_172, out=buf248)
    buf231 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf255 = empty((768, ), device='cpu', dtype=torch.float32)
    buf256 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_70(c_void_p(buf226.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(mul_49.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()))
    buf241 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf239, (768, 2048), (1, 768), 0), view_134, out=buf241)
    buf242 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_71(c_void_p(buf239.data_ptr()), c_void_p(buf242.data_ptr()))
    buf245 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf243, (768, 2048), (1, 768), 0), view_134, out=buf245)
    buf246 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_72(c_void_p(buf243.data_ptr()), c_void_p(buf246.data_ptr()))
    buf249 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf247, (768, 2048), (1, 768), 0), view_134, out=buf249)
    del view_134
    buf250 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf251 = buf226; del buf226  # reuse
    buf252 = buf225; del buf225  # reuse
    buf253 = buf224; del buf224  # reuse
    buf254 = buf251; del buf251  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_73(c_void_p(buf254.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_49.data_ptr()), c_void_p(div_43.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()))
    del div_43
    del mul_49
    buf257 = reinterpret_tensor(buf220, (2048, 3072), (3072, 1), 0); del buf220  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf254, (2048, 768), (768, 1), 0), permute_143, out=buf257)
    buf258 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf254, (768, 2048), (1, 768), 0), view_132, out=buf258)
    del view_132
    buf260 = reinterpret_tensor(buf257, (4, 512, 3072), (1572864, 3072, 1), 0); del buf257  # reuse
    cpp_fused_add_mul_pow_tanh_backward_74(c_void_p(buf260.data_ptr()), c_void_p(addmm_35.data_ptr()), c_void_p(tanh_5.data_ptr()))
    del addmm_35
    del tanh_5
    buf261 = buf248; del buf248  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf260, (2048, 3072), (3072, 1), 0), permute_147, out=buf261)
    buf259 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf267 = empty((768, ), device='cpu', dtype=torch.float32)
    buf268 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_75(c_void_p(buf254.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(mul_43.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()))
    buf262 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf260, (3072, 2048), (1, 3072), 0), view_130, out=buf262)
    del view_130
    buf263 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf264 = buf253; del buf253  # reuse
    buf265 = buf252; del buf252  # reuse
    buf266 = buf254; del buf254  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_76(c_void_p(buf266.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_43.data_ptr()), c_void_p(div_44.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()))
    del div_44
    del mul_43
    buf269 = buf261; del buf261  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf266, (2048, 768), (768, 1), 0), permute_151, out=buf269)
    buf270 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf266, (768, 2048), (1, 768), 0), view_128, out=buf270)
    del view_128
    buf272 = reinterpret_tensor(buf247, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf247  # reuse
    cpp_fused_clone_77(c_void_p(buf269.data_ptr()), c_void_p(buf272.data_ptr()))
    buf273 = reinterpret_tensor(buf269, (48, 512, 64), (32768, 64, 1), 0); del buf269  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_354, reinterpret_tensor(buf272, (48, 512, 64), (32768, 64, 1), 0), out=buf273)
    del permute_354
    buf279 = buf244; del buf244  # reuse
    cpp_fused_view_78(c_void_p(buf273.data_ptr()), c_void_p(buf279.data_ptr()))
    buf280 = reinterpret_tensor(buf273, (2048, 768), (768, 1), 0); del buf273  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf279, permute_164, out=buf280)
    buf274 = reinterpret_tensor(buf236, (48, 512, 512), (262144, 512, 1), 0); del buf236  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf272, (48, 512, 64), (32768, 64, 1), 0), permute_355, out=buf274)
    del permute_355
    buf275 = buf235; del buf235  # reuse
    buf276 = reinterpret_tensor(buf274, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf274  # reuse
    cpp_fused__softmax_backward_data_div_79(c_void_p(buf276.data_ptr()), c_void_p(alias_39.data_ptr()), c_void_p(buf275.data_ptr()))
    del alias_39
    buf277 = reinterpret_tensor(buf272, (48, 64, 512), (32768, 512, 1), 0); del buf272  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_356, reinterpret_tensor(buf276, (48, 512, 512), (262144, 512, 1), 0), out=buf277)
    del permute_356
    buf283 = buf240; del buf240  # reuse
    cpp_fused__unsafe_view_clone_80(c_void_p(buf277.data_ptr()), c_void_p(buf283.data_ptr()))
    buf284 = reinterpret_tensor(buf277, (2048, 768), (768, 1), 0); del buf277  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf283, permute_168, out=buf284)
    buf278 = reinterpret_tensor(buf243, (48, 512, 64), (32768, 64, 1), 0); del buf243  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf276, (48, 512, 512), (262144, 512, 1), 0), permute_357, out=buf278)
    del permute_357
    buf287 = buf239; del buf239  # reuse
    cpp_fused_view_81(c_void_p(buf278.data_ptr()), c_void_p(buf287.data_ptr()))
    buf288 = reinterpret_tensor(buf278, (2048, 768), (768, 1), 0); del buf278  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf287, permute_172, out=buf288)
    buf271 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf295 = empty((768, ), device='cpu', dtype=torch.float32)
    buf296 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_82(c_void_p(buf266.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(mul_41.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()))
    buf281 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf279, (768, 2048), (1, 768), 0), view_112, out=buf281)
    buf282 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_83(c_void_p(buf279.data_ptr()), c_void_p(buf282.data_ptr()))
    buf285 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf283, (768, 2048), (1, 768), 0), view_112, out=buf285)
    buf286 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_84(c_void_p(buf283.data_ptr()), c_void_p(buf286.data_ptr()))
    buf289 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf287, (768, 2048), (1, 768), 0), view_112, out=buf289)
    del view_112
    buf290 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf291 = buf266; del buf266  # reuse
    buf292 = buf265; del buf265  # reuse
    buf293 = buf264; del buf264  # reuse
    buf294 = buf291; del buf291  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_85(c_void_p(buf294.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_41.data_ptr()), c_void_p(div_46.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()))
    del div_46
    del mul_41
    buf297 = reinterpret_tensor(buf260, (2048, 3072), (3072, 1), 0); del buf260  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf294, (2048, 768), (768, 1), 0), permute_143, out=buf297)
    buf298 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf294, (768, 2048), (1, 768), 0), view_110, out=buf298)
    del view_110
    buf300 = reinterpret_tensor(buf297, (4, 512, 3072), (1572864, 3072, 1), 0); del buf297  # reuse
    cpp_fused_add_mul_pow_tanh_backward_86(c_void_p(buf300.data_ptr()), c_void_p(addmm_29.data_ptr()), c_void_p(tanh_4.data_ptr()))
    del addmm_29
    del tanh_4
    buf301 = buf288; del buf288  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf300, (2048, 3072), (3072, 1), 0), permute_147, out=buf301)
    buf299 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf307 = empty((768, ), device='cpu', dtype=torch.float32)
    buf308 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_87(c_void_p(buf294.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(mul_35.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()))
    buf302 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf300, (3072, 2048), (1, 3072), 0), view_108, out=buf302)
    del view_108
    buf303 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf304 = buf293; del buf293  # reuse
    buf305 = buf292; del buf292  # reuse
    buf306 = buf294; del buf294  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_88(c_void_p(buf306.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_35.data_ptr()), c_void_p(div_47.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()))
    del div_47
    del mul_35
    buf309 = buf301; del buf301  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf306, (2048, 768), (768, 1), 0), permute_151, out=buf309)
    buf310 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf306, (768, 2048), (1, 768), 0), view_106, out=buf310)
    del view_106
    buf312 = reinterpret_tensor(buf287, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf287  # reuse
    cpp_fused_clone_89(c_void_p(buf309.data_ptr()), c_void_p(buf312.data_ptr()))
    buf313 = reinterpret_tensor(buf309, (48, 512, 64), (32768, 64, 1), 0); del buf309  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_387, reinterpret_tensor(buf312, (48, 512, 64), (32768, 64, 1), 0), out=buf313)
    del permute_387
    buf319 = buf284; del buf284  # reuse
    cpp_fused_view_90(c_void_p(buf313.data_ptr()), c_void_p(buf319.data_ptr()))
    buf320 = reinterpret_tensor(buf313, (2048, 768), (768, 1), 0); del buf313  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf319, permute_164, out=buf320)
    buf314 = reinterpret_tensor(buf276, (48, 512, 512), (262144, 512, 1), 0); del buf276  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf312, (48, 512, 64), (32768, 64, 1), 0), permute_388, out=buf314)
    del permute_388
    buf315 = buf275; del buf275  # reuse
    buf316 = reinterpret_tensor(buf314, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf314  # reuse
    cpp_fused__softmax_backward_data_div_91(c_void_p(buf316.data_ptr()), c_void_p(alias_41.data_ptr()), c_void_p(buf315.data_ptr()))
    del alias_41
    buf317 = reinterpret_tensor(buf312, (48, 64, 512), (32768, 512, 1), 0); del buf312  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_389, reinterpret_tensor(buf316, (48, 512, 512), (262144, 512, 1), 0), out=buf317)
    del permute_389
    buf323 = buf280; del buf280  # reuse
    cpp_fused__unsafe_view_clone_92(c_void_p(buf317.data_ptr()), c_void_p(buf323.data_ptr()))
    buf324 = reinterpret_tensor(buf317, (2048, 768), (768, 1), 0); del buf317  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf323, permute_168, out=buf324)
    buf318 = reinterpret_tensor(buf283, (48, 512, 64), (32768, 64, 1), 0); del buf283  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf316, (48, 512, 512), (262144, 512, 1), 0), permute_390, out=buf318)
    del permute_390
    buf327 = buf279; del buf279  # reuse
    cpp_fused_view_93(c_void_p(buf318.data_ptr()), c_void_p(buf327.data_ptr()))
    buf328 = reinterpret_tensor(buf318, (2048, 768), (768, 1), 0); del buf318  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf327, permute_172, out=buf328)
    buf311 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf335 = empty((768, ), device='cpu', dtype=torch.float32)
    buf336 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_94(c_void_p(buf306.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(mul_33.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()))
    buf321 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf319, (768, 2048), (1, 768), 0), view_90, out=buf321)
    buf322 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_95(c_void_p(buf319.data_ptr()), c_void_p(buf322.data_ptr()))
    buf325 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf323, (768, 2048), (1, 768), 0), view_90, out=buf325)
    buf326 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_96(c_void_p(buf323.data_ptr()), c_void_p(buf326.data_ptr()))
    buf329 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf327, (768, 2048), (1, 768), 0), view_90, out=buf329)
    del view_90
    buf330 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf331 = buf306; del buf306  # reuse
    buf332 = buf305; del buf305  # reuse
    buf333 = buf304; del buf304  # reuse
    buf334 = buf331; del buf331  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_97(c_void_p(buf334.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_33.data_ptr()), c_void_p(div_49.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()))
    del div_49
    del mul_33
    buf339 = reinterpret_tensor(buf300, (2048, 3072), (3072, 1), 0); del buf300  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf334, (2048, 768), (768, 1), 0), permute_143, out=buf339)
    buf344 = reinterpret_tensor(buf339, (4, 512, 3072), (1572864, 3072, 1), 0); del buf339  # reuse
    cpp_fused_add_mul_pow_tanh_backward_98(c_void_p(buf344.data_ptr()), c_void_p(addmm_23.data_ptr()), c_void_p(tanh_3.data_ptr()))
    del addmm_23
    del tanh_3
    buf345 = buf328; del buf328  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf344, (2048, 3072), (3072, 1), 0), permute_147, out=buf345)
    buf350 = buf333; del buf333  # reuse
    buf351 = buf332; del buf332  # reuse
    buf352 = reinterpret_tensor(buf327, (4, 512, 768), (393216, 768, 1), 0); del buf327  # reuse
    cpp_fused_add_native_layer_norm_backward_99(c_void_p(buf334.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_27.data_ptr()), c_void_p(div_50.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()))
    del div_50
    buf357 = buf324; del buf324  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf352, (2048, 768), (768, 1), 0), permute_151, out=buf357)
    buf362 = reinterpret_tensor(buf320, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf320  # reuse
    cpp_fused_clone_100(c_void_p(buf357.data_ptr()), c_void_p(buf362.data_ptr()))
    buf363 = reinterpret_tensor(buf357, (48, 512, 64), (32768, 64, 1), 0); del buf357  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_420, reinterpret_tensor(buf362, (48, 512, 64), (32768, 64, 1), 0), out=buf363)
    del permute_420
    buf369 = buf323; del buf323  # reuse
    cpp_fused_view_101(c_void_p(buf363.data_ptr()), c_void_p(buf369.data_ptr()))
    buf370 = reinterpret_tensor(buf363, (2048, 768), (768, 1), 0); del buf363  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf369, permute_164, out=buf370)
    buf364 = reinterpret_tensor(buf316, (48, 512, 512), (262144, 512, 1), 0); del buf316  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf362, (48, 512, 64), (32768, 64, 1), 0), permute_421, out=buf364)
    del permute_421
    buf365 = buf315; del buf315  # reuse
    buf366 = reinterpret_tensor(buf364, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf364  # reuse
    cpp_fused__softmax_backward_data_div_102(c_void_p(buf366.data_ptr()), c_void_p(alias_43.data_ptr()), c_void_p(buf365.data_ptr()))
    del alias_43
    buf367 = reinterpret_tensor(buf362, (48, 64, 512), (32768, 512, 1), 0); del buf362  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_422, reinterpret_tensor(buf366, (48, 512, 512), (262144, 512, 1), 0), out=buf367)
    del permute_422
    buf375 = buf319; del buf319  # reuse
    cpp_fused__unsafe_view_clone_103(c_void_p(buf367.data_ptr()), c_void_p(buf375.data_ptr()))
    buf376 = reinterpret_tensor(buf367, (2048, 768), (768, 1), 0); del buf367  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf375, permute_168, out=buf376)
    buf368 = empty((48, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf366, (48, 512, 512), (262144, 512, 1), 0), permute_423, out=buf368)
    del permute_423
    buf381 = empty((2048, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_104(c_void_p(buf368.data_ptr()), c_void_p(buf381.data_ptr()))
    buf382 = reinterpret_tensor(buf368, (2048, 768), (768, 1), 0); del buf368  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf381, permute_172, out=buf382)
    buf359 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf391 = empty((768, ), device='cpu', dtype=torch.float32)
    buf392 = empty((768, ), device='cpu', dtype=torch.float32)
    buf387 = reinterpret_tensor(buf370, (4, 512, 768), (393216, 768, 1), 0); del buf370  # reuse
    buf388 = buf351; del buf351  # reuse
    buf389 = buf350; del buf350  # reuse
    buf390 = buf387; del buf387  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_105(c_void_p(buf390.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(mul_25.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(div_52.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()))
    del div_52
    del mul_25
    buf393 = empty((2048, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf390, (2048, 768), (768, 1), 0), permute_143, out=buf393)
    buf396 = reinterpret_tensor(buf393, (4, 512, 3072), (1572864, 3072, 1), 0); del buf393  # reuse
    cpp_fused_add_mul_pow_tanh_backward_106(c_void_p(buf396.data_ptr()), c_void_p(addmm_17.data_ptr()), c_void_p(tanh_2.data_ptr()))
    del addmm_17
    del tanh_2
    buf397 = buf382; del buf382  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf396, (2048, 3072), (3072, 1), 0), permute_147, out=buf397)
    buf400 = buf389; del buf389  # reuse
    buf401 = buf388; del buf388  # reuse
    buf402 = reinterpret_tensor(buf376, (4, 512, 768), (393216, 768, 1), 0); del buf376  # reuse
    cpp_fused_add_native_layer_norm_backward_107(c_void_p(buf390.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_19.data_ptr()), c_void_p(div_53.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf402.data_ptr()))
    del div_53
    buf405 = empty((2048, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf402, (2048, 768), (768, 1), 0), permute_151, out=buf405)
    buf408 = empty((4, 12, 512, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_108(c_void_p(buf405.data_ptr()), c_void_p(buf408.data_ptr()))
    buf409 = reinterpret_tensor(buf405, (48, 512, 64), (32768, 64, 1), 0); del buf405  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_453, reinterpret_tensor(buf408, (48, 512, 64), (32768, 64, 1), 0), out=buf409)
    del permute_453
    buf415 = empty((2048, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_109(c_void_p(buf409.data_ptr()), c_void_p(buf415.data_ptr()))
    buf416 = reinterpret_tensor(buf409, (2048, 768), (768, 1), 0); del buf409  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf415, permute_164, out=buf416)
    buf410 = reinterpret_tensor(buf366, (48, 512, 512), (262144, 512, 1), 0); del buf366  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf408, (48, 512, 64), (32768, 64, 1), 0), permute_454, out=buf410)
    del permute_454
    buf411 = buf365; del buf365  # reuse
    buf412 = reinterpret_tensor(buf410, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf410  # reuse
    cpp_fused__softmax_backward_data_div_110(c_void_p(buf412.data_ptr()), c_void_p(alias_45.data_ptr()), c_void_p(buf411.data_ptr()))
    del alias_45
    buf413 = reinterpret_tensor(buf408, (48, 64, 512), (32768, 512, 1), 0); del buf408  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_455, reinterpret_tensor(buf412, (48, 512, 512), (262144, 512, 1), 0), out=buf413)
    del permute_455
    buf419 = empty((2048, 768), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_111(c_void_p(buf413.data_ptr()), c_void_p(buf419.data_ptr()))
    buf420 = reinterpret_tensor(buf413, (2048, 768), (768, 1), 0); del buf413  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf419, permute_168, out=buf420)
    buf414 = empty((48, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf412, (48, 512, 512), (262144, 512, 1), 0), permute_456, out=buf414)
    del permute_456
    buf423 = empty((2048, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_112(c_void_p(buf414.data_ptr()), c_void_p(buf423.data_ptr()))
    buf424 = reinterpret_tensor(buf414, (2048, 768), (768, 1), 0); del buf414  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf423, permute_172, out=buf424)
    buf407 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf431 = empty((768, ), device='cpu', dtype=torch.float32)
    buf432 = empty((768, ), device='cpu', dtype=torch.float32)
    buf427 = reinterpret_tensor(buf416, (4, 512, 768), (393216, 768, 1), 0); del buf416  # reuse
    buf428 = buf401; del buf401  # reuse
    buf429 = buf400; del buf400  # reuse
    buf430 = buf427; del buf427  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_113(c_void_p(buf430.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(mul_17.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(div_55.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()))
    del div_55
    del mul_17
    buf433 = empty((2048, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf430, (2048, 768), (768, 1), 0), permute_143, out=buf433)
    buf436 = reinterpret_tensor(buf433, (4, 512, 3072), (1572864, 3072, 1), 0); del buf433  # reuse
    cpp_fused_add_mul_pow_tanh_backward_114(c_void_p(buf436.data_ptr()), c_void_p(addmm_11.data_ptr()), c_void_p(tanh_1.data_ptr()))
    del addmm_11
    del tanh_1
    buf437 = buf424; del buf424  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf436, (2048, 3072), (3072, 1), 0), permute_147, out=buf437)
    buf440 = buf429; del buf429  # reuse
    buf441 = buf428; del buf428  # reuse
    buf442 = reinterpret_tensor(buf420, (4, 512, 768), (393216, 768, 1), 0); del buf420  # reuse
    cpp_fused_add_native_layer_norm_backward_115(c_void_p(buf430.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_11.data_ptr()), c_void_p(div_56.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()))
    del div_56
    buf445 = empty((2048, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf442, (2048, 768), (768, 1), 0), permute_151, out=buf445)
    buf448 = empty((4, 12, 512, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_116(c_void_p(buf445.data_ptr()), c_void_p(buf448.data_ptr()))
    buf449 = reinterpret_tensor(buf445, (48, 512, 64), (32768, 64, 1), 0); del buf445  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_486, reinterpret_tensor(buf448, (48, 512, 64), (32768, 64, 1), 0), out=buf449)
    del permute_486
    buf455 = empty((2048, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_117(c_void_p(buf449.data_ptr()), c_void_p(buf455.data_ptr()))
    buf456 = reinterpret_tensor(buf449, (2048, 768), (768, 1), 0); del buf449  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf455, permute_164, out=buf456)
    buf450 = reinterpret_tensor(buf412, (48, 512, 512), (262144, 512, 1), 0); del buf412  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf448, (48, 512, 64), (32768, 64, 1), 0), permute_487, out=buf450)
    del permute_487
    buf451 = buf411; del buf411  # reuse
    buf452 = reinterpret_tensor(buf450, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf450  # reuse
    cpp_fused__softmax_backward_data_div_118(c_void_p(buf452.data_ptr()), c_void_p(alias_47.data_ptr()), c_void_p(buf451.data_ptr()))
    del alias_47
    buf453 = reinterpret_tensor(buf448, (48, 64, 512), (32768, 512, 1), 0); del buf448  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_488, reinterpret_tensor(buf452, (48, 512, 512), (262144, 512, 1), 0), out=buf453)
    del permute_488
    buf459 = empty((2048, 768), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_119(c_void_p(buf453.data_ptr()), c_void_p(buf459.data_ptr()))
    buf460 = reinterpret_tensor(buf453, (2048, 768), (768, 1), 0); del buf453  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf459, permute_168, out=buf460)
    buf454 = empty((48, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf452, (48, 512, 512), (262144, 512, 1), 0), permute_489, out=buf454)
    del permute_489
    buf463 = empty((2048, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_120(c_void_p(buf454.data_ptr()), c_void_p(buf463.data_ptr()))
    buf464 = reinterpret_tensor(buf454, (2048, 768), (768, 1), 0); del buf454  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf463, permute_172, out=buf464)
    buf447 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf471 = empty((768, ), device='cpu', dtype=torch.float32)
    buf472 = empty((768, ), device='cpu', dtype=torch.float32)
    buf337 = buf135; del buf135  # reuse
    buf473 = buf337; del buf337  # reuse
    buf338 = buf136; del buf136  # reuse
    buf474 = buf338; del buf338  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_121(c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(mul_9.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf472.data_ptr()))
    del buf15
    del buf16
    del buf175
    del buf176
    del buf215
    del buf216
    del buf255
    del buf256
    del buf295
    del buf296
    buf340 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf334, (768, 2048), (1, 768), 0), view_88, out=buf340)
    del view_88
    buf341 = reinterpret_tensor(buf96, (1, 768), (768, 1), 0); del buf96  # reuse
    buf353 = buf95; del buf95  # reuse
    buf354 = buf56; del buf56  # reuse
    buf395 = reinterpret_tensor(buf55, (1, 768), (768, 1), 0); del buf55  # reuse
    buf403 = buf472; del buf472  # reuse
    buf404 = buf471; del buf471  # reuse
    buf435 = reinterpret_tensor(buf432, (1, 768), (768, 1), 0); del buf432  # reuse
    buf443 = buf431; del buf431  # reuse
    buf444 = buf392; del buf392  # reuse
    buf467 = reinterpret_tensor(buf456, (4, 512, 768), (393216, 768, 1), 0); del buf456  # reuse
    buf468 = buf441; del buf441  # reuse
    buf469 = buf440; del buf440  # reuse
    buf470 = buf467; del buf467  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_122(c_void_p(buf470.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(mul_27.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(mul_19.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(mul_11.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_9.data_ptr()), c_void_p(div_58.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()))
    del buf334
    del buf345
    del buf397
    del buf437
    del buf460
    del div_58
    del mul_11
    del mul_19
    del mul_27
    del mul_9
    del primals_22
    buf475 = empty((2048, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf470, (2048, 768), (768, 1), 0), permute_143, out=buf475)
    del permute_143
    buf480 = reinterpret_tensor(buf475, (4, 512, 3072), (1572864, 3072, 1), 0); del buf475  # reuse
    cpp_fused_add_mul_pow_tanh_backward_123(c_void_p(buf480.data_ptr()), c_void_p(addmm_5.data_ptr()), c_void_p(tanh.data_ptr()))
    del addmm_5
    del tanh
    buf481 = buf464; del buf464  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf480, (2048, 3072), (3072, 1), 0), permute_147, out=buf481)
    del permute_147
    buf477 = reinterpret_tensor(buf391, (1, 768), (768, 1), 0); del buf391  # reuse
    buf489 = buf336; del buf336  # reuse
    buf490 = buf335; del buf335  # reuse
    buf342 = reinterpret_tensor(buf139, (768, ), (1, ), 0); del buf139  # reuse
    buf478 = buf342; del buf342  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_124(c_void_p(buf478.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(mul_3.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf490.data_ptr()))
    del buf179
    del buf19
    del buf219
    del buf259
    del buf299
    del buf341
    del buf395
    del buf435
    del buf477
    del buf59
    del buf99
    buf394 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf390, (768, 2048), (1, 768), 0), view_66, out=buf394)
    del buf390
    del view_66
    buf434 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf430, (768, 2048), (1, 768), 0), view_44, out=buf434)
    del buf430
    del view_44
    buf476 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf470, (768, 2048), (1, 768), 0), view_22, out=buf476)
    del view_22
    buf343 = buf138; del buf138  # reuse
    buf479 = buf343; del buf343  # reuse
    cpp_fused_add_125(c_void_p(buf479.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf476.data_ptr()))
    del buf178
    del buf18
    del buf218
    del buf258
    del buf298
    del buf340
    del buf394
    buf346 = reinterpret_tensor(buf98, (3072, 768), (768, 1), 0); del buf98  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf344, (3072, 2048), (1, 3072), 0), view_86, out=buf346)
    del view_86
    buf347 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf399 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf439 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf483 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf348 = reinterpret_tensor(buf103, (3072, ), (1, ), 0); del buf103  # reuse
    buf484 = buf348; del buf348  # reuse
    cpp_fused_add_sum_126(c_void_p(buf484.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf483.data_ptr()))
    del buf143
    del buf183
    del buf223
    del buf23
    del buf263
    del buf303
    del buf344
    del buf347
    del buf399
    del buf439
    del buf483
    del buf63
    buf398 = reinterpret_tensor(buf58, (3072, 768), (768, 1), 0); del buf58  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf396, (3072, 2048), (1, 3072), 0), view_64, out=buf398)
    del buf396
    del view_64
    buf438 = reinterpret_tensor(buf476, (3072, 768), (768, 1), 0); del buf476  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf436, (3072, 2048), (1, 3072), 0), view_42, out=buf438)
    del buf436
    del view_42
    buf482 = reinterpret_tensor(buf434, (3072, 768), (768, 1), 0); del buf434  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf480, (3072, 2048), (1, 3072), 0), view_20, out=buf482)
    del buf480
    del view_20
    buf349 = buf102; del buf102  # reuse
    buf485 = buf349; del buf349  # reuse
    buf355 = buf107; del buf107  # reuse
    buf491 = buf355; del buf355  # reuse
    buf356 = buf108; del buf108  # reuse
    buf492 = buf356; del buf356  # reuse
    cpp_fused_add_127(c_void_p(buf485.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf490.data_ptr()))
    del buf142
    del buf147
    del buf148
    del buf182
    del buf187
    del buf188
    del buf22
    del buf222
    del buf227
    del buf228
    del buf262
    del buf267
    del buf268
    del buf27
    del buf28
    del buf302
    del buf307
    del buf308
    del buf346
    del buf353
    del buf354
    del buf398
    del buf403
    del buf404
    del buf438
    del buf443
    del buf444
    del buf482
    del buf489
    del buf490
    del buf62
    del buf67
    buf358 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf352, (768, 2048), (1, 768), 0), view_84, out=buf358)
    del buf352
    del view_84
    buf486 = buf469; del buf469  # reuse
    buf487 = buf468; del buf468  # reuse
    buf488 = buf470; del buf470  # reuse
    buf495 = reinterpret_tensor(buf68, (1, 768), (768, 1), 0); del buf68  # reuse
    buf360 = reinterpret_tensor(buf111, (768, ), (1, ), 0); del buf111  # reuse
    buf496 = buf360; del buf360  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_128(c_void_p(buf488.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_3.data_ptr()), c_void_p(div_59.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf495.data_ptr()))
    del buf151
    del buf191
    del buf231
    del buf271
    del buf31
    del buf311
    del buf359
    del buf481
    del div_59
    del mul_3
    del primals_16
    buf406 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf402, (768, 2048), (1, 768), 0), view_62, out=buf406)
    del view_62
    buf446 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf442, (768, 2048), (1, 768), 0), view_40, out=buf446)
    del view_40
    buf494 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf488, (768, 2048), (1, 768), 0), view_18, out=buf494)
    del view_18
    buf361 = buf110; del buf110  # reuse
    buf497 = buf361; del buf361  # reuse
    cpp_fused_add_129(c_void_p(buf497.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf494.data_ptr()))
    del buf150
    del buf190
    del buf230
    del buf270
    del buf30
    del buf310
    del buf358
    buf371 = buf70; del buf70  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf369, (768, 2048), (1, 768), 0), view_68, out=buf371)
    buf372 = buf71; del buf71  # reuse
    buf418 = buf495; del buf495  # reuse
    buf458 = buf447; del buf447  # reuse
    cpp_fused_sum_130(c_void_p(buf369.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf458.data_ptr()))
    buf493 = buf369; del buf369  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf488, (2048, 768), (768, 1), 0), permute_151, out=buf493)
    del permute_151
    buf498 = reinterpret_tensor(buf442, (4, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf442  # reuse
    cpp_fused_clone_131(c_void_p(buf493.data_ptr()), c_void_p(buf498.data_ptr()))
    buf499 = reinterpret_tensor(buf493, (48, 512, 64), (32768, 64, 1), 0); del buf493  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_519, reinterpret_tensor(buf498, (48, 512, 64), (32768, 64, 1), 0), out=buf499)
    del permute_519
    buf505 = reinterpret_tensor(buf402, (2048, 768), (768, 1), 0); del buf402  # reuse
    buf508 = buf407; del buf407  # reuse
    buf373 = reinterpret_tensor(buf122, (768, ), (1, ), 0); del buf122  # reuse
    buf509 = buf373; del buf373  # reuse
    cpp_fused_add_sum_view_132(c_void_p(buf509.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf508.data_ptr()))
    del buf162
    del buf202
    del buf242
    del buf282
    del buf322
    del buf372
    del buf418
    del buf499
    buf417 = buf494; del buf494  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf415, (768, 2048), (1, 768), 0), view_46, out=buf417)
    del buf415
    buf457 = buf446; del buf446  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf455, (768, 2048), (1, 768), 0), view_24, out=buf457)
    del buf455
    buf507 = buf406; del buf406  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf505, (768, 2048), (1, 768), 0), view_2, out=buf507)
    buf374 = buf121; del buf121  # reuse
    buf510 = buf374; del buf374  # reuse
    cpp_fused_add_133(c_void_p(buf510.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf507.data_ptr()))
    del buf161
    del buf201
    del buf241
    del buf281
    del buf321
    del buf371
    del buf41
    buf377 = buf81; del buf81  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf375, (768, 2048), (1, 768), 0), view_68, out=buf377)
    buf378 = buf82; del buf82  # reuse
    buf422 = buf508; del buf508  # reuse
    buf462 = buf458; del buf458  # reuse
    cpp_fused_sum_134(c_void_p(buf375.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf462.data_ptr()))
    buf500 = reinterpret_tensor(buf452, (48, 512, 512), (262144, 512, 1), 0); del buf452  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf498, (48, 512, 64), (32768, 64, 1), 0), permute_520, out=buf500)
    del permute_520
    buf501 = buf451; del buf451  # reuse
    buf502 = reinterpret_tensor(buf500, (4, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf500  # reuse
    cpp_fused__softmax_backward_data_div_135(c_void_p(buf502.data_ptr()), c_void_p(alias_49.data_ptr()), c_void_p(buf501.data_ptr()))
    del alias_49
    del buf501
    buf503 = reinterpret_tensor(buf498, (48, 64, 512), (32768, 512, 1), 0); del buf498  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_521, reinterpret_tensor(buf502, (48, 512, 512), (262144, 512, 1), 0), out=buf503)
    del permute_521
    buf511 = buf375; del buf375  # reuse
    buf514 = buf42; del buf42  # reuse
    buf379 = reinterpret_tensor(buf126, (768, ), (1, ), 0); del buf126  # reuse
    buf515 = buf379; del buf379  # reuse
    cpp_fused__unsafe_view_add_clone_sum_136(c_void_p(buf515.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf514.data_ptr()))
    del buf166
    del buf206
    del buf246
    del buf286
    del buf326
    del buf378
    del buf422
    del buf503
    buf421 = buf507; del buf507  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf419, (768, 2048), (1, 768), 0), view_46, out=buf421)
    del buf419
    buf461 = buf457; del buf457  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf459, (768, 2048), (1, 768), 0), view_24, out=buf461)
    buf513 = buf417; del buf417  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf511, (768, 2048), (1, 768), 0), view_2, out=buf513)
    buf380 = buf125; del buf125  # reuse
    buf516 = buf380; del buf380  # reuse
    cpp_fused_add_137(c_void_p(buf516.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf513.data_ptr()))
    del buf165
    del buf205
    del buf245
    del buf285
    del buf325
    del buf377
    del buf421
    buf383 = buf85; del buf85  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf381, (768, 2048), (1, 768), 0), view_68, out=buf383)
    del view_68
    buf384 = buf86; del buf86  # reuse
    buf426 = buf514; del buf514  # reuse
    buf466 = buf462; del buf462  # reuse
    cpp_fused_sum_138(c_void_p(buf381.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf466.data_ptr()))
    buf504 = reinterpret_tensor(buf381, (48, 512, 64), (32768, 64, 1), 0); del buf381  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf502, (48, 512, 512), (262144, 512, 1), 0), permute_522, out=buf504)
    del buf502
    del permute_522
    buf517 = buf459; del buf459  # reuse
    buf520 = buf46; del buf46  # reuse
    buf385 = reinterpret_tensor(buf130, (768, ), (1, ), 0); del buf130  # reuse
    buf521 = buf385; del buf385  # reuse
    cpp_fused_add_sum_view_139(c_void_p(buf521.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf520.data_ptr()))
    del buf170
    del buf210
    del buf250
    del buf290
    del buf330
    del buf384
    del buf426
    del buf466
    del buf50
    del buf504
    del buf520
    buf425 = buf513; del buf513  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf423, (768, 2048), (1, 768), 0), view_46, out=buf425)
    del buf423
    del view_46
    buf465 = buf461; del buf461  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf463, (768, 2048), (1, 768), 0), view_24, out=buf465)
    del view_24
    buf519 = buf45; del buf45  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf517, (768, 2048), (1, 768), 0), view_2, out=buf519)
    del view_2
    buf386 = buf129; del buf129  # reuse
    buf522 = buf386; del buf386  # reuse
    cpp_fused_add_140(c_void_p(buf522.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf519.data_ptr()))
    del buf169
    del buf209
    del buf249
    del buf289
    del buf329
    del buf383
    del buf425
    del buf465
    del buf49
    del buf519
    del buf89
    buf506 = buf463; del buf463  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf505, permute_164, out=buf506)
    del permute_164
    buf512 = buf505; del buf505  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf511, permute_168, out=buf512)
    del permute_168
    buf518 = buf511; del buf511  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf517, permute_172, out=buf518)
    del buf517
    del permute_172
    buf523 = buf488; del buf488  # reuse
    cpp_fused_add_141(c_void_p(buf523.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf518.data_ptr()))
    del buf506
    del buf512
    del buf518
    buf524 = reinterpret_tensor(buf8, (2048, 128), (128, 1), 0); del buf8  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf523, (2048, 768), (768, 1), 0), permute_539, out=buf524)
    del permute_539
    buf525 = empty((768, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf523, (768, 2048), (1, 768), 0), view, out=buf525)
    del view
    buf526 = buf90; del buf90  # reuse
    buf527 = buf487; del buf487  # reuse
    buf528 = buf486; del buf486  # reuse
    buf529 = reinterpret_tensor(buf0, (4, 512, 128), (65536, 128, 1), 0); del buf0  # reuse
    buf541 = empty((4, 512, 128), device='cpu', dtype=torch.float32)
    buf530 = empty((128, ), device='cpu', dtype=torch.float32)
    buf531 = empty((128, ), device='cpu', dtype=torch.float32)
    buf532 = empty((512, 128), device='cpu', dtype=torch.float32)
    buf533 = empty((1, 512, 128), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_native_layer_norm_backward_sum_142(c_void_p(buf523.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(mul_1.data_ptr()), c_void_p(div_61.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(slice_2.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(buf533.data_ptr()))
    del buf523
    del buf524
    del buf527
    del buf528
    del div_61
    del mul_1
    del primals_4
    aten.index_put_(buf532, [slice_2], buf533, True)
    del buf533
    del slice_2
    buf536 = empty((2, 128), device='cpu', dtype=torch.float32)
    buf537 = buf529; del buf529  # reuse
    cpp_fused_embedding_dense_backward_143(c_void_p(buf537.data_ptr()), c_void_p(expand.data_ptr()), c_void_p(buf536.data_ptr()))
    aten.index_put_(buf536, [expand], buf537, True)
    del buf537
    del expand
    buf540 = empty((30000, 128), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_144(c_void_p(buf540.data_ptr()))
    aten.index_put_(buf540, [primals_32], buf541, True)
    del buf541
    del primals_32
    return (buf540, buf536, buf532, buf530, buf531, reinterpret_tensor(buf525, (768, 128), (128, 1), 0), reinterpret_tensor(buf526, (768, ), (1, ), 0), buf522, buf521, buf516, buf515, buf510, buf509, buf497, buf496, buf491, buf492, buf485, buf484, buf479, buf478, buf473, buf474, reinterpret_tensor(buf10, (128, 768), (768, 1), 0), reinterpret_tensor(buf11, (128, ), (1, ), 0), buf6, buf7, reinterpret_tensor(buf1, (30000, 128), (128, 1), 0), reinterpret_tensor(buf2, (30000, ), (1, ), 0), None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((4, 512), (512, 1), device='cpu', dtype=torch.int64)
    expand = rand_strided((4, 512), (0, 1), device='cpu', dtype=torch.int64)
    slice_2 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    mul_1 = rand_strided((4, 512, 128), (65536, 128, 1), device='cpu', dtype=torch.float32)
    view = rand_strided((2048, 128), (128, 1), device='cpu', dtype=torch.float32)
    view_2 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_18 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_3 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_20 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_5 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_22 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_9 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_24 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_40 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_11 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_42 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_11 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_1 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_44 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_17 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_46 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_62 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_19 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_64 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_17 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_2 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_66 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_25 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_68 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_84 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_27 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_86 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_23 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_3 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_88 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_33 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_90 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_106 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_35 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_108 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_29 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_4 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_110 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_41 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_112 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_128 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_43 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_130 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_35 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_5 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_132 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_49 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_134 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_150 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_51 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_152 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_41 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_6 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_154 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_57 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_156 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_172 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_59 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_174 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_47 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_7 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_176 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_65 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_178 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_194 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_67 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_196 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_53 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_8 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_198 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_73 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_200 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_216 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_75 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_218 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_59 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_9 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_220 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_81 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_222 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_238 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_83 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_240 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_65 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_10 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_242 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_89 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_244 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_260 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_91 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_262 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_71 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_11 = rand_strided((4, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    view_264 = rand_strided((2048, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_97 = rand_strided((4, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_266 = rand_strided((2048, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_73 = rand_strided((2048, 128), (128, 1), device='cpu', dtype=torch.float32)
    tanh_12 = rand_strided((4, 512, 128), (65536, 128, 1), device='cpu', dtype=torch.float32)
    getitem_51 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_25 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_268 = rand_strided((2048, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_135 = rand_strided((30000, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_139 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_25 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_143 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_147 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_26 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_151 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_156 = rand_strided((48, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_157 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_27 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_158 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_159 = rand_strided((48, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_164 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_168 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_172 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_28 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_29 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_189 = rand_strided((48, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_190 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_29 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_191 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_192 = rand_strided((48, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    div_31 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_32 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_222 = rand_strided((48, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_223 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_31 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_224 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_225 = rand_strided((48, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    div_34 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_35 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_255 = rand_strided((48, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_256 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_33 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_257 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_258 = rand_strided((48, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    div_37 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_38 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_288 = rand_strided((48, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_289 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_35 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_290 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_291 = rand_strided((48, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    div_40 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_41 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_321 = rand_strided((48, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_322 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_37 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_323 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_324 = rand_strided((48, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    div_43 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_44 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_354 = rand_strided((48, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_355 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_39 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_356 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_357 = rand_strided((48, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    div_46 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_47 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_387 = rand_strided((48, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_388 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_41 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_389 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_390 = rand_strided((48, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    div_49 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_50 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_420 = rand_strided((48, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_421 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_43 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_422 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_423 = rand_strided((48, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    div_52 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_53 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_453 = rand_strided((48, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_454 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_45 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_455 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_456 = rand_strided((48, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    div_55 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_56 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_486 = rand_strided((48, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_487 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_47 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_488 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_489 = rand_strided((48, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    div_58 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_59 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_519 = rand_strided((48, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_520 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_49 = rand_strided((4, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_521 = rand_strided((48, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_522 = rand_strided((48, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_539 = rand_strided((768, 128), (128, 1), device='cpu', dtype=torch.float32)
    div_61 = rand_strided((4, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((4, 512, 30000), (15360000, 30000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_16, primals_22, primals_26, primals_32, expand, slice_2, mul_1, view, view_2, view_18, mul_3, view_20, addmm_5, tanh, view_22, mul_9, view_24, view_40, mul_11, view_42, addmm_11, tanh_1, view_44, mul_17, view_46, view_62, mul_19, view_64, addmm_17, tanh_2, view_66, mul_25, view_68, view_84, mul_27, view_86, addmm_23, tanh_3, view_88, mul_33, view_90, view_106, mul_35, view_108, addmm_29, tanh_4, view_110, mul_41, view_112, view_128, mul_43, view_130, addmm_35, tanh_5, view_132, mul_49, view_134, view_150, mul_51, view_152, addmm_41, tanh_6, view_154, mul_57, view_156, view_172, mul_59, view_174, addmm_47, tanh_7, view_176, mul_65, view_178, view_194, mul_67, view_196, addmm_53, tanh_8, view_198, mul_73, view_200, view_216, mul_75, view_218, addmm_59, tanh_9, view_220, mul_81, view_222, view_238, mul_83, view_240, addmm_65, tanh_10, view_242, mul_89, view_244, view_260, mul_91, view_262, addmm_71, tanh_11, view_264, mul_97, view_266, addmm_73, tanh_12, getitem_51, rsqrt_25, view_268, permute_135, permute_139, div_25, permute_143, permute_147, div_26, permute_151, permute_156, permute_157, alias_27, permute_158, permute_159, permute_164, permute_168, permute_172, div_28, div_29, permute_189, permute_190, alias_29, permute_191, permute_192, div_31, div_32, permute_222, permute_223, alias_31, permute_224, permute_225, div_34, div_35, permute_255, permute_256, alias_33, permute_257, permute_258, div_37, div_38, permute_288, permute_289, alias_35, permute_290, permute_291, div_40, div_41, permute_321, permute_322, alias_37, permute_323, permute_324, div_43, div_44, permute_354, permute_355, alias_39, permute_356, permute_357, div_46, div_47, permute_387, permute_388, alias_41, permute_389, permute_390, div_49, div_50, permute_420, permute_421, alias_43, permute_422, permute_423, div_52, div_53, permute_453, permute_454, alias_45, permute_455, permute_456, div_55, div_56, permute_486, permute_487, alias_47, permute_488, permute_489, div_58, div_59, permute_519, permute_520, alias_49, permute_521, permute_522, permute_539, div_61, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_Albert', benchmark_compiled_module)
