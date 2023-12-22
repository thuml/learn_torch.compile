
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


cpp_fused_native_layer_norm_backward_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_1 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_backward_sum_2 = async_compile.cpp('''
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
                       float* out_ptr4)
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        auto tmp11 = static_cast<float>(8.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
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
                            auto tmp13 = in_ptr1[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + (512L*(static_cast<long>(x2) % static_cast<long>(768L))) + (393216L*x0))];
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
                            auto tmp21 = in_ptr3[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp22 = in_ptr4[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp25 = tmp11 ? tmp16 : tmp24;
                        auto tmp26 = tmp4 ? tmp7 : tmp25;
                        out_ptr0[static_cast<long>(x2 + (2304L*x1) + (1179648L*x0))] = tmp26;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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


cpp_fused_add_native_layer_norm_backward_sum_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
    }
}
''')


cpp_fused_clone_sum_9 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        auto tmp11 = static_cast<float>(8.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
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
                            auto tmp13 = in_ptr1[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + (512L*(static_cast<long>(x2) % static_cast<long>(768L))) + (393216L*x0))];
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
                            auto tmp21 = in_ptr3[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp22 = in_ptr4[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp25 = tmp11 ? tmp16 : tmp24;
                        auto tmp26 = tmp4 ? tmp7 : tmp25;
                        out_ptr0[static_cast<long>(x2 + (2304L*x1) + (1179648L*x0))] = tmp26;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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


cpp_fused_add_native_layer_norm_backward_sum_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
    }
}
''')


cpp_fused_clone_sum_15 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        auto tmp11 = static_cast<float>(8.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
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
                            auto tmp13 = in_ptr1[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + (512L*(static_cast<long>(x2) % static_cast<long>(768L))) + (393216L*x0))];
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
                            auto tmp21 = in_ptr3[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp22 = in_ptr4[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp25 = tmp11 ? tmp16 : tmp24;
                        auto tmp26 = tmp4 ? tmp7 : tmp25;
                        out_ptr0[static_cast<long>(x2 + (2304L*x1) + (1179648L*x0))] = tmp26;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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


cpp_fused_add_native_layer_norm_backward_sum_20 = async_compile.cpp('''
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
                       float* out_ptr4)
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
    }
}
''')


cpp_fused_clone_sum_21 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        auto tmp11 = static_cast<float>(8.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
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
                            auto tmp13 = in_ptr1[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + (512L*(static_cast<long>(x2) % static_cast<long>(768L))) + (393216L*x0))];
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
                            auto tmp21 = in_ptr3[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp22 = in_ptr4[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp25 = tmp11 ? tmp16 : tmp24;
                        auto tmp26 = tmp4 ? tmp7 : tmp25;
                        out_ptr0[static_cast<long>(x2 + (2304L*x1) + (1179648L*x0))] = tmp26;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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


cpp_fused_add_native_layer_norm_backward_sum_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
    }
}
''')


cpp_fused_clone_sum_27 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        auto tmp11 = static_cast<float>(8.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
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
                            auto tmp13 = in_ptr1[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + (512L*(static_cast<long>(x2) % static_cast<long>(768L))) + (393216L*x0))];
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
                            auto tmp21 = in_ptr3[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp22 = in_ptr4[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp25 = tmp11 ? tmp16 : tmp24;
                        auto tmp26 = tmp4 ? tmp7 : tmp25;
                        out_ptr0[static_cast<long>(x2 + (2304L*x1) + (1179648L*x0))] = tmp26;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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


cpp_fused_add_native_layer_norm_backward_sum_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
    }
}
''')


cpp_fused_clone_sum_33 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        auto tmp11 = static_cast<float>(8.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
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
                            auto tmp13 = in_ptr1[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + (512L*(static_cast<long>(x2) % static_cast<long>(768L))) + (393216L*x0))];
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
                            auto tmp21 = in_ptr3[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp22 = in_ptr4[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp25 = tmp11 ? tmp16 : tmp24;
                        auto tmp26 = tmp4 ? tmp7 : tmp25;
                        out_ptr0[static_cast<long>(x2 + (2304L*x1) + (1179648L*x0))] = tmp26;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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


cpp_fused_add_native_layer_norm_backward_sum_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
    }
}
''')


cpp_fused_clone_sum_39 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        auto tmp11 = static_cast<float>(8.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
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
                            auto tmp13 = in_ptr1[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + (512L*(static_cast<long>(x2) % static_cast<long>(768L))) + (393216L*x0))];
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
                            auto tmp21 = in_ptr3[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp22 = in_ptr4[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp25 = tmp11 ? tmp16 : tmp24;
                        auto tmp26 = tmp4 ? tmp7 : tmp25;
                        out_ptr0[static_cast<long>(x2 + (2304L*x1) + (1179648L*x0))] = tmp26;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_43 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_backward_sum_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
    }
}
''')


cpp_fused_clone_sum_45 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        auto tmp11 = static_cast<float>(8.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
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
                            auto tmp13 = in_ptr1[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + (512L*(static_cast<long>(x2) % static_cast<long>(768L))) + (393216L*x0))];
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
                            auto tmp21 = in_ptr3[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp22 = in_ptr4[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp25 = tmp11 ? tmp16 : tmp24;
                        auto tmp26 = tmp4 ? tmp7 : tmp25;
                        out_ptr0[static_cast<long>(x2 + (2304L*x1) + (1179648L*x0))] = tmp26;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_49 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_backward_sum_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
    }
}
''')


cpp_fused_clone_sum_51 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        auto tmp11 = static_cast<float>(8.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
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
                            auto tmp13 = in_ptr1[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + (512L*(static_cast<long>(x2) % static_cast<long>(768L))) + (393216L*x0))];
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
                            auto tmp21 = in_ptr3[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp22 = in_ptr4[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp25 = tmp11 ? tmp16 : tmp24;
                        auto tmp26 = tmp4 ? tmp7 : tmp25;
                        out_ptr0[static_cast<long>(x2 + (2304L*x1) + (1179648L*x0))] = tmp26;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_55 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_backward_sum_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
    }
}
''')


cpp_fused_clone_sum_57 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        auto tmp11 = static_cast<float>(8.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
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
                            auto tmp13 = in_ptr1[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + (512L*(static_cast<long>(x2) % static_cast<long>(768L))) + (393216L*x0))];
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
                            auto tmp21 = in_ptr3[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp22 = in_ptr4[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp25 = tmp11 ? tmp16 : tmp24;
                        auto tmp26 = tmp4 ? tmp7 : tmp25;
                        out_ptr0[static_cast<long>(x2 + (2304L*x1) + (1179648L*x0))] = tmp26;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_61 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_backward_sum_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
    }
}
''')


cpp_fused_clone_sum_63 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        auto tmp11 = static_cast<float>(8.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
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
                            auto tmp13 = in_ptr1[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + (512L*(static_cast<long>(x2) % static_cast<long>(768L))) + (393216L*x0))];
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
                            auto tmp21 = in_ptr3[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp22 = in_ptr4[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp25 = tmp11 ? tmp16 : tmp24;
                        auto tmp26 = tmp4 ? tmp7 : tmp25;
                        out_ptr0[static_cast<long>(x2 + (2304L*x1) + (1179648L*x0))] = tmp26;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
    }
}
''')


cpp_fused_add_mul_pow_sum_tanh_backward_67 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_backward_sum_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (393216L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (32768L*x1) + (393216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp7 = tmp3 - tmp6;
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = decltype(tmp7)::blendv(tmp9, tmp7, tmp0);
                        auto tmp11 = static_cast<float>(8.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (262144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2304L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(768);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
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
                            auto tmp13 = in_ptr1[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp14 = in_ptr2[static_cast<long>(x1 + (512L*(static_cast<long>(x2) % static_cast<long>(768L))) + (393216L*x0))];
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
                            auto tmp21 = in_ptr3[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp22 = in_ptr4[static_cast<long>((64L*x1) + (32768L*(static_cast<long>(c10::div_floor_integer(x2, 64L)) % static_cast<long>(12L))) + (393216L*x0) + (static_cast<long>(x2) % static_cast<long>(64L)))];
                            auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                            return tmp23;
                        }
                        ;
                        auto tmp24 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp25 = tmp11 ? tmp16 : tmp24;
                        auto tmp26 = tmp4 ? tmp7 : tmp25;
                        out_ptr0[static_cast<long>(x2 + (2304L*x1) + (1179648L*x0))] = tmp26;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_dense_backward_native_layer_norm_backward_scalar_tensor_sum_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const long* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
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
                    auto tmp19 = in_ptr5[static_cast<long>(x0)];
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
                    auto tmp20 = static_cast<int>(-1);
                    auto tmp21 = tmp19 == tmp20;
                    auto tmp22 = static_cast<float>(0.0);
                    auto tmp23 = to_float_mask(tmp21);
                    auto tmp24 = at::vec::Vectorized<float>(tmp22);
                    auto tmp25 = decltype(tmp24)::blendv(tmp18, tmp24, tmp23);
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp25.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_out_ptr0[static_cast<long>(393216L + x0)];
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                auto tmp3 = static_cast<bool>(0);
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = tmp3 ? tmp4 : tmp2;
                out_ptr7[static_cast<long>(x0)] = tmp5;
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_73 = async_compile.cpp('''
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
    primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, view, view_1, mul, slice_4, mul_2, addmm_2, tanh, mul_8, slice_8, mul_10, addmm_6, tanh_1, mul_16, slice_12, mul_18, addmm_10, tanh_2, mul_24, slice_16, mul_26, addmm_14, tanh_3, mul_32, slice_20, mul_34, addmm_18, tanh_4, mul_40, slice_24, mul_42, addmm_22, tanh_5, mul_48, slice_28, mul_50, addmm_26, tanh_6, mul_56, slice_32, mul_58, addmm_30, tanh_7, mul_64, slice_36, mul_66, addmm_34, tanh_8, mul_72, slice_40, mul_74, addmm_38, tanh_9, mul_80, slice_44, mul_82, addmm_42, tanh_10, mul_88, slice_48, mul_90, addmm_46, tanh_11, mul_96, view_219, permute_63, div_24, permute_65, permute_66, permute_67, permute_68, div_25, permute_69, permute_70, permute_72, permute_73, alias_25, permute_74, permute_75, permute_80, permute_81, div_27, permute_82, permute_83, permute_84, permute_85, div_28, permute_86, permute_87, permute_89, permute_90, alias_27, permute_91, permute_92, permute_97, permute_98, div_30, permute_99, permute_100, permute_101, permute_102, div_31, permute_103, permute_104, permute_106, permute_107, alias_29, permute_108, permute_109, permute_114, permute_115, div_33, permute_116, permute_117, permute_118, permute_119, div_34, permute_120, permute_121, permute_123, permute_124, alias_31, permute_125, permute_126, permute_131, permute_132, div_36, permute_133, permute_134, permute_135, permute_136, div_37, permute_137, permute_138, permute_140, permute_141, alias_33, permute_142, permute_143, permute_148, permute_149, div_39, permute_150, permute_151, permute_152, permute_153, div_40, permute_154, permute_155, permute_157, permute_158, alias_35, permute_159, permute_160, permute_165, permute_166, div_42, permute_167, permute_168, permute_169, permute_170, div_43, permute_171, permute_172, permute_174, permute_175, alias_37, permute_176, permute_177, permute_182, permute_183, div_45, permute_184, permute_185, permute_186, permute_187, div_46, permute_188, permute_189, permute_191, permute_192, alias_39, permute_193, permute_194, permute_199, permute_200, div_48, permute_201, permute_202, permute_203, permute_204, div_49, permute_205, permute_206, permute_208, permute_209, alias_41, permute_210, permute_211, permute_216, permute_217, div_51, permute_218, permute_219, permute_220, permute_221, div_52, permute_222, permute_223, permute_225, permute_226, alias_43, permute_227, permute_228, permute_233, permute_234, div_54, permute_235, permute_236, permute_237, permute_238, div_55, permute_239, permute_240, permute_242, permute_243, alias_45, permute_244, permute_245, permute_250, permute_251, div_57, permute_252, permute_253, permute_254, permute_255, div_58, permute_256, permute_257, permute_259, permute_260, alias_47, permute_261, permute_262, permute_267, permute_268, div_60, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25 = args
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
    assert_size_stride(view, (2, 512), (512, 1))
    assert_size_stride(view_1, (1, 512), (512, 1))
    assert_size_stride(mul, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_4, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_2, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_2, (1024, 3072), (3072, 1))
    assert_size_stride(tanh, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_8, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_8, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_10, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_6, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_1, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_16, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_12, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_18, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_10, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_2, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_24, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_16, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_26, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_14, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_3, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_32, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_20, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_34, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_18, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_4, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_40, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_24, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_42, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_22, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_5, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_48, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_28, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_50, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_26, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_6, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_56, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_32, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_58, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_30, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_7, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_64, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_36, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_66, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_34, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_8, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_72, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_40, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_74, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_38, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_9, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_80, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_44, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_82, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_42, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_10, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_88, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_48, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_90, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_46, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_11, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_96, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(view_219, (1024, 768), (768, 1))
    assert_size_stride(permute_63, (50257, 768), (768, 1))
    assert_size_stride(div_24, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_65, (768, 3072), (1, 768))
    assert_size_stride(permute_66, (3072, 1024), (1, 3072))
    assert_size_stride(permute_67, (3072, 768), (1, 3072))
    assert_size_stride(permute_68, (768, 1024), (1, 768))
    assert_size_stride(div_25, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_69, (768, 768), (1, 768))
    assert_size_stride(permute_70, (768, 1024), (1, 768))
    assert_size_stride(permute_72, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_73, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_25, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_74, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_75, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_80, (2304, 768), (1, 2304))
    assert_size_stride(permute_81, (768, 1024), (1, 768))
    assert_size_stride(div_27, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_82, (768, 3072), (1, 768))
    assert_size_stride(permute_83, (3072, 1024), (1, 3072))
    assert_size_stride(permute_84, (3072, 768), (1, 3072))
    assert_size_stride(permute_85, (768, 1024), (1, 768))
    assert_size_stride(div_28, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_86, (768, 768), (1, 768))
    assert_size_stride(permute_87, (768, 1024), (1, 768))
    assert_size_stride(permute_89, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_90, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_27, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_91, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_92, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_97, (2304, 768), (1, 2304))
    assert_size_stride(permute_98, (768, 1024), (1, 768))
    assert_size_stride(div_30, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_99, (768, 3072), (1, 768))
    assert_size_stride(permute_100, (3072, 1024), (1, 3072))
    assert_size_stride(permute_101, (3072, 768), (1, 3072))
    assert_size_stride(permute_102, (768, 1024), (1, 768))
    assert_size_stride(div_31, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_103, (768, 768), (1, 768))
    assert_size_stride(permute_104, (768, 1024), (1, 768))
    assert_size_stride(permute_106, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_107, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_29, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_108, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_109, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_114, (2304, 768), (1, 2304))
    assert_size_stride(permute_115, (768, 1024), (1, 768))
    assert_size_stride(div_33, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_116, (768, 3072), (1, 768))
    assert_size_stride(permute_117, (3072, 1024), (1, 3072))
    assert_size_stride(permute_118, (3072, 768), (1, 3072))
    assert_size_stride(permute_119, (768, 1024), (1, 768))
    assert_size_stride(div_34, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_120, (768, 768), (1, 768))
    assert_size_stride(permute_121, (768, 1024), (1, 768))
    assert_size_stride(permute_123, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_124, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_31, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_125, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_126, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_131, (2304, 768), (1, 2304))
    assert_size_stride(permute_132, (768, 1024), (1, 768))
    assert_size_stride(div_36, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_133, (768, 3072), (1, 768))
    assert_size_stride(permute_134, (3072, 1024), (1, 3072))
    assert_size_stride(permute_135, (3072, 768), (1, 3072))
    assert_size_stride(permute_136, (768, 1024), (1, 768))
    assert_size_stride(div_37, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_137, (768, 768), (1, 768))
    assert_size_stride(permute_138, (768, 1024), (1, 768))
    assert_size_stride(permute_140, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_141, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_33, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_142, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_143, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_148, (2304, 768), (1, 2304))
    assert_size_stride(permute_149, (768, 1024), (1, 768))
    assert_size_stride(div_39, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_150, (768, 3072), (1, 768))
    assert_size_stride(permute_151, (3072, 1024), (1, 3072))
    assert_size_stride(permute_152, (3072, 768), (1, 3072))
    assert_size_stride(permute_153, (768, 1024), (1, 768))
    assert_size_stride(div_40, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_154, (768, 768), (1, 768))
    assert_size_stride(permute_155, (768, 1024), (1, 768))
    assert_size_stride(permute_157, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_158, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_35, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_159, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_160, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_165, (2304, 768), (1, 2304))
    assert_size_stride(permute_166, (768, 1024), (1, 768))
    assert_size_stride(div_42, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_167, (768, 3072), (1, 768))
    assert_size_stride(permute_168, (3072, 1024), (1, 3072))
    assert_size_stride(permute_169, (3072, 768), (1, 3072))
    assert_size_stride(permute_170, (768, 1024), (1, 768))
    assert_size_stride(div_43, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_171, (768, 768), (1, 768))
    assert_size_stride(permute_172, (768, 1024), (1, 768))
    assert_size_stride(permute_174, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_175, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_37, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_176, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_177, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_182, (2304, 768), (1, 2304))
    assert_size_stride(permute_183, (768, 1024), (1, 768))
    assert_size_stride(div_45, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_184, (768, 3072), (1, 768))
    assert_size_stride(permute_185, (3072, 1024), (1, 3072))
    assert_size_stride(permute_186, (3072, 768), (1, 3072))
    assert_size_stride(permute_187, (768, 1024), (1, 768))
    assert_size_stride(div_46, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_188, (768, 768), (1, 768))
    assert_size_stride(permute_189, (768, 1024), (1, 768))
    assert_size_stride(permute_191, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_192, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_39, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_193, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_194, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_199, (2304, 768), (1, 2304))
    assert_size_stride(permute_200, (768, 1024), (1, 768))
    assert_size_stride(div_48, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_201, (768, 3072), (1, 768))
    assert_size_stride(permute_202, (3072, 1024), (1, 3072))
    assert_size_stride(permute_203, (3072, 768), (1, 3072))
    assert_size_stride(permute_204, (768, 1024), (1, 768))
    assert_size_stride(div_49, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_205, (768, 768), (1, 768))
    assert_size_stride(permute_206, (768, 1024), (1, 768))
    assert_size_stride(permute_208, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_209, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_41, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_210, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_211, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_216, (2304, 768), (1, 2304))
    assert_size_stride(permute_217, (768, 1024), (1, 768))
    assert_size_stride(div_51, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_218, (768, 3072), (1, 768))
    assert_size_stride(permute_219, (3072, 1024), (1, 3072))
    assert_size_stride(permute_220, (3072, 768), (1, 3072))
    assert_size_stride(permute_221, (768, 1024), (1, 768))
    assert_size_stride(div_52, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_222, (768, 768), (1, 768))
    assert_size_stride(permute_223, (768, 1024), (1, 768))
    assert_size_stride(permute_225, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_226, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_43, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_227, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_228, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_233, (2304, 768), (1, 2304))
    assert_size_stride(permute_234, (768, 1024), (1, 768))
    assert_size_stride(div_54, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_235, (768, 3072), (1, 768))
    assert_size_stride(permute_236, (3072, 1024), (1, 3072))
    assert_size_stride(permute_237, (3072, 768), (1, 3072))
    assert_size_stride(permute_238, (768, 1024), (1, 768))
    assert_size_stride(div_55, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_239, (768, 768), (1, 768))
    assert_size_stride(permute_240, (768, 1024), (1, 768))
    assert_size_stride(permute_242, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_243, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_45, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_244, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_245, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_250, (2304, 768), (1, 2304))
    assert_size_stride(permute_251, (768, 1024), (1, 768))
    assert_size_stride(div_57, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_252, (768, 3072), (1, 768))
    assert_size_stride(permute_253, (3072, 1024), (1, 3072))
    assert_size_stride(permute_254, (3072, 768), (1, 3072))
    assert_size_stride(permute_255, (768, 1024), (1, 768))
    assert_size_stride(div_58, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_256, (768, 768), (1, 768))
    assert_size_stride(permute_257, (768, 1024), (1, 768))
    assert_size_stride(permute_259, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_260, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_47, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_261, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_262, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_267, (2304, 768), (1, 2304))
    assert_size_stride(permute_268, (768, 1024), (1, 768))
    assert_size_stride(div_60, (2, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (2, 512, 50257), (25731584, 50257, 1))
    assert_size_stride(tangents_2, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_3, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_4, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_5, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_6, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_7, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_8, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_9, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_10, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_11, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_12, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_13, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_14, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_15, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_16, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_17, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_18, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_19, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_20, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_21, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_22, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_23, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_24, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_25, (2, 12, 512, 64), (393216, 32768, 64, 1))
    buf0 = empty((50257, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (50257, 1024), (1, 50257), 0), view_219, out=buf0)
    del view_219
    buf1 = empty((1024, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1024, 50257), (50257, 1), 0), permute_63, out=buf1)
    del permute_63
    del tangents_1
    buf2 = empty_strided((2, 512, 1), (512, 1, 1024), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((2, 512, 1), (512, 1, 1024), device='cpu', dtype=torch.float32)
    buf4 = empty((2, 512, 768), device='cpu', dtype=torch.float32)
    buf5 = empty((768, ), device='cpu', dtype=torch.float32)
    buf6 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_backward_0(c_void_p(buf1.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(mul_96.data_ptr()), c_void_p(div_24.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del div_24
    del mul_96
    del primals_147
    buf7 = empty((1024, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf4, (1024, 768), (768, 1), 0), permute_65, out=buf7)
    del permute_65
    buf8 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_66, reinterpret_tensor(buf4, (1024, 768), (768, 1), 0), out=buf8)
    del permute_66
    buf9 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf10 = reinterpret_tensor(buf7, (2, 512, 3072), (1572864, 3072, 1), 0); del buf7  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_1(c_void_p(buf10.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(addmm_46.data_ptr()), c_void_p(tanh_11.data_ptr()), c_void_p(buf9.data_ptr()))
    del addmm_46
    del tanh_11
    buf11 = buf1; del buf1  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf10, (1024, 3072), (3072, 1), 0), permute_67, out=buf11)
    del permute_67
    buf12 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_68, reinterpret_tensor(buf10, (1024, 3072), (3072, 1), 0), out=buf12)
    del permute_68
    buf13 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf14 = buf3; del buf3  # reuse
    buf15 = buf2; del buf2  # reuse
    buf16 = empty((768, ), device='cpu', dtype=torch.float32)
    buf17 = empty((768, ), device='cpu', dtype=torch.float32)
    buf18 = reinterpret_tensor(buf11, (2, 512, 768), (393216, 768, 1), 0); del buf11  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_2(c_void_p(buf18.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(mul_90.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(div_25.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()))
    del div_25
    del mul_90
    del primals_145
    buf19 = reinterpret_tensor(buf4, (1024, 768), (768, 1), 0); del buf4  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf18, (1024, 768), (768, 1), 0), permute_69, out=buf19)
    del permute_69
    buf20 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_70, reinterpret_tensor(buf18, (1024, 768), (768, 1), 0), out=buf20)
    del permute_70
    buf21 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf22 = empty((2, 12, 512, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_sum_3(c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()))
    buf23 = reinterpret_tensor(buf19, (24, 512, 64), (32768, 64, 1), 0); del buf19  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_72, reinterpret_tensor(buf22, (24, 512, 64), (32768, 64, 1), 0), out=buf23)
    del permute_72
    buf24 = empty((24, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf22, (24, 512, 64), (32768, 64, 1), 0), permute_73, out=buf24)
    del permute_73
    buf25 = empty_strided((2, 12, 512, 1), (6144, 512, 1, 12288), device='cpu', dtype=torch.float32)
    buf26 = reinterpret_tensor(buf24, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf24  # reuse
    cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_4(c_void_p(buf26.data_ptr()), c_void_p(alias_25.data_ptr()), c_void_p(slice_48.data_ptr()), c_void_p(buf25.data_ptr()))
    del alias_25
    del slice_48
    buf27 = reinterpret_tensor(buf22, (24, 64, 512), (32768, 512, 1), 0); del buf22  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_74, reinterpret_tensor(buf26, (24, 512, 512), (262144, 512, 1), 0), out=buf27)
    del permute_74
    buf28 = empty((24, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf26, (24, 512, 512), (262144, 512, 1), 0), permute_75, out=buf28)
    del permute_75
    buf29 = empty((2, 512, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_5(c_void_p(buf28.data_ptr()), c_void_p(tangents_24.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(tangents_25.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf29.data_ptr()))
    del tangents_24
    del tangents_25
    buf30 = reinterpret_tensor(buf28, (1024, 768), (768, 1), 0); del buf28  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf29, (1024, 2304), (2304, 1), 0), permute_80, out=buf30)
    del permute_80
    buf31 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_81, reinterpret_tensor(buf29, (1024, 2304), (2304, 1), 0), out=buf31)
    del permute_81
    buf32 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf33 = buf15; del buf15  # reuse
    buf34 = buf14; del buf14  # reuse
    buf35 = empty((768, ), device='cpu', dtype=torch.float32)
    buf36 = empty((768, ), device='cpu', dtype=torch.float32)
    buf37 = buf18; del buf18  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_6(c_void_p(buf37.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(mul_88.data_ptr()), c_void_p(div_27.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()))
    del div_27
    del mul_88
    del primals_143
    buf38 = reinterpret_tensor(buf10, (1024, 3072), (3072, 1), 0); del buf10  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf37, (1024, 768), (768, 1), 0), permute_82, out=buf38)
    del permute_82
    buf39 = reinterpret_tensor(buf29, (3072, 768), (768, 1), 0); del buf29  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_83, reinterpret_tensor(buf37, (1024, 768), (768, 1), 0), out=buf39)
    del permute_83
    buf40 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf41 = reinterpret_tensor(buf38, (2, 512, 3072), (1572864, 3072, 1), 0); del buf38  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_7(c_void_p(buf41.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(addmm_42.data_ptr()), c_void_p(tanh_10.data_ptr()), c_void_p(buf40.data_ptr()))
    del addmm_42
    del tanh_10
    buf42 = buf30; del buf30  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf41, (1024, 3072), (3072, 1), 0), permute_84, out=buf42)
    del permute_84
    buf43 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_85, reinterpret_tensor(buf41, (1024, 3072), (3072, 1), 0), out=buf43)
    del permute_85
    buf44 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf45 = buf34; del buf34  # reuse
    buf46 = buf33; del buf33  # reuse
    buf47 = empty((768, ), device='cpu', dtype=torch.float32)
    buf48 = empty((768, ), device='cpu', dtype=torch.float32)
    buf49 = buf37; del buf37  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_8(c_void_p(buf49.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(mul_82.data_ptr()), c_void_p(div_28.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    del div_28
    del mul_82
    del primals_141
    buf50 = buf42; del buf42  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf49, (1024, 768), (768, 1), 0), permute_86, out=buf50)
    del permute_86
    buf51 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_87, reinterpret_tensor(buf49, (1024, 768), (768, 1), 0), out=buf51)
    del permute_87
    buf52 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf53 = reinterpret_tensor(buf27, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf27  # reuse
    cpp_fused_clone_sum_9(c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()))
    buf54 = reinterpret_tensor(buf50, (24, 512, 64), (32768, 64, 1), 0); del buf50  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_89, reinterpret_tensor(buf53, (24, 512, 64), (32768, 64, 1), 0), out=buf54)
    del permute_89
    buf55 = reinterpret_tensor(buf26, (24, 512, 512), (262144, 512, 1), 0); del buf26  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf53, (24, 512, 64), (32768, 64, 1), 0), permute_90, out=buf55)
    del permute_90
    buf56 = buf25; del buf25  # reuse
    buf57 = reinterpret_tensor(buf55, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf55  # reuse
    cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_10(c_void_p(buf57.data_ptr()), c_void_p(alias_27.data_ptr()), c_void_p(slice_44.data_ptr()), c_void_p(buf56.data_ptr()))
    del alias_27
    del slice_44
    buf58 = reinterpret_tensor(buf53, (24, 64, 512), (32768, 512, 1), 0); del buf53  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_91, reinterpret_tensor(buf57, (24, 512, 512), (262144, 512, 1), 0), out=buf58)
    del permute_91
    buf59 = buf23; del buf23  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf57, (24, 512, 512), (262144, 512, 1), 0), permute_92, out=buf59)
    del permute_92
    buf60 = empty((2, 512, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_11(c_void_p(buf59.data_ptr()), c_void_p(tangents_22.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(tangents_23.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf60.data_ptr()))
    del tangents_22
    del tangents_23
    buf61 = reinterpret_tensor(buf59, (1024, 768), (768, 1), 0); del buf59  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf60, (1024, 2304), (2304, 1), 0), permute_97, out=buf61)
    del permute_97
    buf62 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_98, reinterpret_tensor(buf60, (1024, 2304), (2304, 1), 0), out=buf62)
    del permute_98
    buf63 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf64 = buf46; del buf46  # reuse
    buf65 = buf45; del buf45  # reuse
    buf66 = empty((768, ), device='cpu', dtype=torch.float32)
    buf67 = empty((768, ), device='cpu', dtype=torch.float32)
    buf68 = buf49; del buf49  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_12(c_void_p(buf68.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(mul_80.data_ptr()), c_void_p(div_30.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()))
    del div_30
    del mul_80
    del primals_139
    buf69 = reinterpret_tensor(buf41, (1024, 3072), (3072, 1), 0); del buf41  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf68, (1024, 768), (768, 1), 0), permute_99, out=buf69)
    del permute_99
    buf70 = reinterpret_tensor(buf60, (3072, 768), (768, 1), 0); del buf60  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_100, reinterpret_tensor(buf68, (1024, 768), (768, 1), 0), out=buf70)
    del permute_100
    buf71 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf72 = reinterpret_tensor(buf69, (2, 512, 3072), (1572864, 3072, 1), 0); del buf69  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_13(c_void_p(buf72.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(addmm_38.data_ptr()), c_void_p(tanh_9.data_ptr()), c_void_p(buf71.data_ptr()))
    del addmm_38
    del tanh_9
    buf73 = buf61; del buf61  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf72, (1024, 3072), (3072, 1), 0), permute_101, out=buf73)
    del permute_101
    buf74 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_102, reinterpret_tensor(buf72, (1024, 3072), (3072, 1), 0), out=buf74)
    del permute_102
    buf75 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf76 = buf65; del buf65  # reuse
    buf77 = buf64; del buf64  # reuse
    buf78 = empty((768, ), device='cpu', dtype=torch.float32)
    buf79 = empty((768, ), device='cpu', dtype=torch.float32)
    buf80 = buf68; del buf68  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_14(c_void_p(buf80.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(mul_74.data_ptr()), c_void_p(div_31.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    del div_31
    del mul_74
    del primals_137
    buf81 = buf73; del buf73  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf80, (1024, 768), (768, 1), 0), permute_103, out=buf81)
    del permute_103
    buf82 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_104, reinterpret_tensor(buf80, (1024, 768), (768, 1), 0), out=buf82)
    del permute_104
    buf83 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf84 = reinterpret_tensor(buf58, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf58  # reuse
    cpp_fused_clone_sum_15(c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()))
    buf85 = reinterpret_tensor(buf81, (24, 512, 64), (32768, 64, 1), 0); del buf81  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_106, reinterpret_tensor(buf84, (24, 512, 64), (32768, 64, 1), 0), out=buf85)
    del permute_106
    buf86 = reinterpret_tensor(buf57, (24, 512, 512), (262144, 512, 1), 0); del buf57  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf84, (24, 512, 64), (32768, 64, 1), 0), permute_107, out=buf86)
    del permute_107
    buf87 = buf56; del buf56  # reuse
    buf88 = reinterpret_tensor(buf86, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf86  # reuse
    cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_16(c_void_p(buf88.data_ptr()), c_void_p(alias_29.data_ptr()), c_void_p(slice_40.data_ptr()), c_void_p(buf87.data_ptr()))
    del alias_29
    del slice_40
    buf89 = reinterpret_tensor(buf84, (24, 64, 512), (32768, 512, 1), 0); del buf84  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_108, reinterpret_tensor(buf88, (24, 512, 512), (262144, 512, 1), 0), out=buf89)
    del permute_108
    buf90 = buf54; del buf54  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf88, (24, 512, 512), (262144, 512, 1), 0), permute_109, out=buf90)
    del permute_109
    buf91 = empty((2, 512, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_17(c_void_p(buf90.data_ptr()), c_void_p(tangents_20.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(tangents_21.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf91.data_ptr()))
    del tangents_20
    del tangents_21
    buf92 = reinterpret_tensor(buf90, (1024, 768), (768, 1), 0); del buf90  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf91, (1024, 2304), (2304, 1), 0), permute_114, out=buf92)
    del permute_114
    buf93 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_115, reinterpret_tensor(buf91, (1024, 2304), (2304, 1), 0), out=buf93)
    del permute_115
    buf94 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf95 = buf77; del buf77  # reuse
    buf96 = buf76; del buf76  # reuse
    buf97 = empty((768, ), device='cpu', dtype=torch.float32)
    buf98 = empty((768, ), device='cpu', dtype=torch.float32)
    buf99 = buf80; del buf80  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_18(c_void_p(buf99.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(mul_72.data_ptr()), c_void_p(div_33.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()))
    del div_33
    del mul_72
    del primals_135
    buf100 = reinterpret_tensor(buf72, (1024, 3072), (3072, 1), 0); del buf72  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf99, (1024, 768), (768, 1), 0), permute_116, out=buf100)
    del permute_116
    buf101 = reinterpret_tensor(buf91, (3072, 768), (768, 1), 0); del buf91  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_117, reinterpret_tensor(buf99, (1024, 768), (768, 1), 0), out=buf101)
    del permute_117
    buf102 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf103 = reinterpret_tensor(buf100, (2, 512, 3072), (1572864, 3072, 1), 0); del buf100  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_19(c_void_p(buf103.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(tanh_8.data_ptr()), c_void_p(buf102.data_ptr()))
    del addmm_34
    del tanh_8
    buf104 = buf92; del buf92  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf103, (1024, 3072), (3072, 1), 0), permute_118, out=buf104)
    del permute_118
    buf105 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_119, reinterpret_tensor(buf103, (1024, 3072), (3072, 1), 0), out=buf105)
    del permute_119
    buf106 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf107 = buf96; del buf96  # reuse
    buf108 = buf95; del buf95  # reuse
    buf109 = empty((768, ), device='cpu', dtype=torch.float32)
    buf110 = empty((768, ), device='cpu', dtype=torch.float32)
    buf111 = reinterpret_tensor(buf104, (2, 512, 768), (393216, 768, 1), 0); del buf104  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_20(c_void_p(buf111.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(mul_66.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(div_34.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()))
    del div_34
    del mul_66
    del primals_133
    buf112 = reinterpret_tensor(buf99, (1024, 768), (768, 1), 0); del buf99  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf111, (1024, 768), (768, 1), 0), permute_120, out=buf112)
    del permute_120
    buf113 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_121, reinterpret_tensor(buf111, (1024, 768), (768, 1), 0), out=buf113)
    del permute_121
    buf114 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf115 = reinterpret_tensor(buf89, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf89  # reuse
    cpp_fused_clone_sum_21(c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()))
    buf116 = reinterpret_tensor(buf112, (24, 512, 64), (32768, 64, 1), 0); del buf112  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_123, reinterpret_tensor(buf115, (24, 512, 64), (32768, 64, 1), 0), out=buf116)
    del permute_123
    buf117 = reinterpret_tensor(buf88, (24, 512, 512), (262144, 512, 1), 0); del buf88  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf115, (24, 512, 64), (32768, 64, 1), 0), permute_124, out=buf117)
    del permute_124
    buf118 = buf87; del buf87  # reuse
    buf119 = reinterpret_tensor(buf117, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf117  # reuse
    cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_22(c_void_p(buf119.data_ptr()), c_void_p(alias_31.data_ptr()), c_void_p(slice_36.data_ptr()), c_void_p(buf118.data_ptr()))
    del alias_31
    del slice_36
    buf120 = reinterpret_tensor(buf115, (24, 64, 512), (32768, 512, 1), 0); del buf115  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_125, reinterpret_tensor(buf119, (24, 512, 512), (262144, 512, 1), 0), out=buf120)
    del permute_125
    buf121 = buf85; del buf85  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf119, (24, 512, 512), (262144, 512, 1), 0), permute_126, out=buf121)
    del permute_126
    buf122 = empty((2, 512, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_23(c_void_p(buf121.data_ptr()), c_void_p(tangents_18.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(tangents_19.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf122.data_ptr()))
    del tangents_18
    del tangents_19
    buf123 = reinterpret_tensor(buf121, (1024, 768), (768, 1), 0); del buf121  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf122, (1024, 2304), (2304, 1), 0), permute_131, out=buf123)
    del permute_131
    buf124 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_132, reinterpret_tensor(buf122, (1024, 2304), (2304, 1), 0), out=buf124)
    del permute_132
    buf125 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf126 = buf108; del buf108  # reuse
    buf127 = buf107; del buf107  # reuse
    buf128 = empty((768, ), device='cpu', dtype=torch.float32)
    buf129 = empty((768, ), device='cpu', dtype=torch.float32)
    buf130 = buf111; del buf111  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_24(c_void_p(buf130.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(mul_64.data_ptr()), c_void_p(div_36.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()))
    del div_36
    del mul_64
    del primals_131
    buf131 = reinterpret_tensor(buf103, (1024, 3072), (3072, 1), 0); del buf103  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf130, (1024, 768), (768, 1), 0), permute_133, out=buf131)
    del permute_133
    buf132 = reinterpret_tensor(buf122, (3072, 768), (768, 1), 0); del buf122  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_134, reinterpret_tensor(buf130, (1024, 768), (768, 1), 0), out=buf132)
    del permute_134
    buf133 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf134 = reinterpret_tensor(buf131, (2, 512, 3072), (1572864, 3072, 1), 0); del buf131  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_25(c_void_p(buf134.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(addmm_30.data_ptr()), c_void_p(tanh_7.data_ptr()), c_void_p(buf133.data_ptr()))
    del addmm_30
    del tanh_7
    buf135 = buf123; del buf123  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf134, (1024, 3072), (3072, 1), 0), permute_135, out=buf135)
    del permute_135
    buf136 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_136, reinterpret_tensor(buf134, (1024, 3072), (3072, 1), 0), out=buf136)
    del permute_136
    buf137 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf138 = buf127; del buf127  # reuse
    buf139 = buf126; del buf126  # reuse
    buf140 = empty((768, ), device='cpu', dtype=torch.float32)
    buf141 = empty((768, ), device='cpu', dtype=torch.float32)
    buf142 = buf130; del buf130  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_26(c_void_p(buf142.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(mul_58.data_ptr()), c_void_p(div_37.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()))
    del div_37
    del mul_58
    del primals_129
    buf143 = buf135; del buf135  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf142, (1024, 768), (768, 1), 0), permute_137, out=buf143)
    del permute_137
    buf144 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_138, reinterpret_tensor(buf142, (1024, 768), (768, 1), 0), out=buf144)
    del permute_138
    buf145 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf146 = reinterpret_tensor(buf120, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf120  # reuse
    cpp_fused_clone_sum_27(c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()))
    buf147 = reinterpret_tensor(buf143, (24, 512, 64), (32768, 64, 1), 0); del buf143  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_140, reinterpret_tensor(buf146, (24, 512, 64), (32768, 64, 1), 0), out=buf147)
    del permute_140
    buf148 = reinterpret_tensor(buf119, (24, 512, 512), (262144, 512, 1), 0); del buf119  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf146, (24, 512, 64), (32768, 64, 1), 0), permute_141, out=buf148)
    del permute_141
    buf149 = buf118; del buf118  # reuse
    buf150 = reinterpret_tensor(buf148, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf148  # reuse
    cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_28(c_void_p(buf150.data_ptr()), c_void_p(alias_33.data_ptr()), c_void_p(slice_32.data_ptr()), c_void_p(buf149.data_ptr()))
    del alias_33
    del slice_32
    buf151 = reinterpret_tensor(buf146, (24, 64, 512), (32768, 512, 1), 0); del buf146  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_142, reinterpret_tensor(buf150, (24, 512, 512), (262144, 512, 1), 0), out=buf151)
    del permute_142
    buf152 = buf116; del buf116  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf150, (24, 512, 512), (262144, 512, 1), 0), permute_143, out=buf152)
    del permute_143
    buf153 = empty((2, 512, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_29(c_void_p(buf152.data_ptr()), c_void_p(tangents_16.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(tangents_17.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf153.data_ptr()))
    del tangents_16
    del tangents_17
    buf154 = reinterpret_tensor(buf152, (1024, 768), (768, 1), 0); del buf152  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf153, (1024, 2304), (2304, 1), 0), permute_148, out=buf154)
    del permute_148
    buf155 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_149, reinterpret_tensor(buf153, (1024, 2304), (2304, 1), 0), out=buf155)
    del permute_149
    buf156 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf157 = buf139; del buf139  # reuse
    buf158 = buf138; del buf138  # reuse
    buf159 = empty((768, ), device='cpu', dtype=torch.float32)
    buf160 = empty((768, ), device='cpu', dtype=torch.float32)
    buf161 = buf142; del buf142  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_30(c_void_p(buf161.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(mul_56.data_ptr()), c_void_p(div_39.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()))
    del div_39
    del mul_56
    del primals_127
    buf162 = reinterpret_tensor(buf134, (1024, 3072), (3072, 1), 0); del buf134  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf161, (1024, 768), (768, 1), 0), permute_150, out=buf162)
    del permute_150
    buf163 = reinterpret_tensor(buf153, (3072, 768), (768, 1), 0); del buf153  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_151, reinterpret_tensor(buf161, (1024, 768), (768, 1), 0), out=buf163)
    del permute_151
    buf164 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf165 = reinterpret_tensor(buf162, (2, 512, 3072), (1572864, 3072, 1), 0); del buf162  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_31(c_void_p(buf165.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(addmm_26.data_ptr()), c_void_p(tanh_6.data_ptr()), c_void_p(buf164.data_ptr()))
    del addmm_26
    del tanh_6
    buf166 = buf154; del buf154  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf165, (1024, 3072), (3072, 1), 0), permute_152, out=buf166)
    del permute_152
    buf167 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_153, reinterpret_tensor(buf165, (1024, 3072), (3072, 1), 0), out=buf167)
    del permute_153
    buf168 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf169 = buf158; del buf158  # reuse
    buf170 = buf157; del buf157  # reuse
    buf171 = empty((768, ), device='cpu', dtype=torch.float32)
    buf172 = empty((768, ), device='cpu', dtype=torch.float32)
    buf173 = buf161; del buf161  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_32(c_void_p(buf173.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(mul_50.data_ptr()), c_void_p(div_40.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()))
    del div_40
    del mul_50
    del primals_125
    buf174 = buf166; del buf166  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf173, (1024, 768), (768, 1), 0), permute_154, out=buf174)
    del permute_154
    buf175 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_155, reinterpret_tensor(buf173, (1024, 768), (768, 1), 0), out=buf175)
    del permute_155
    buf176 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf177 = reinterpret_tensor(buf151, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf151  # reuse
    cpp_fused_clone_sum_33(c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()))
    buf178 = reinterpret_tensor(buf174, (24, 512, 64), (32768, 64, 1), 0); del buf174  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_157, reinterpret_tensor(buf177, (24, 512, 64), (32768, 64, 1), 0), out=buf178)
    del permute_157
    buf179 = reinterpret_tensor(buf150, (24, 512, 512), (262144, 512, 1), 0); del buf150  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf177, (24, 512, 64), (32768, 64, 1), 0), permute_158, out=buf179)
    del permute_158
    buf180 = buf149; del buf149  # reuse
    buf181 = reinterpret_tensor(buf179, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf179  # reuse
    cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_34(c_void_p(buf181.data_ptr()), c_void_p(alias_35.data_ptr()), c_void_p(slice_28.data_ptr()), c_void_p(buf180.data_ptr()))
    del alias_35
    del slice_28
    buf182 = reinterpret_tensor(buf177, (24, 64, 512), (32768, 512, 1), 0); del buf177  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_159, reinterpret_tensor(buf181, (24, 512, 512), (262144, 512, 1), 0), out=buf182)
    del permute_159
    buf183 = buf147; del buf147  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf181, (24, 512, 512), (262144, 512, 1), 0), permute_160, out=buf183)
    del permute_160
    buf184 = empty((2, 512, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_35(c_void_p(buf183.data_ptr()), c_void_p(tangents_14.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(tangents_15.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf184.data_ptr()))
    del tangents_14
    del tangents_15
    buf185 = reinterpret_tensor(buf183, (1024, 768), (768, 1), 0); del buf183  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf184, (1024, 2304), (2304, 1), 0), permute_165, out=buf185)
    del permute_165
    buf186 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_166, reinterpret_tensor(buf184, (1024, 2304), (2304, 1), 0), out=buf186)
    del permute_166
    buf187 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf188 = buf170; del buf170  # reuse
    buf189 = buf169; del buf169  # reuse
    buf190 = empty((768, ), device='cpu', dtype=torch.float32)
    buf191 = empty((768, ), device='cpu', dtype=torch.float32)
    buf192 = buf173; del buf173  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_36(c_void_p(buf192.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(mul_48.data_ptr()), c_void_p(div_42.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()))
    del div_42
    del mul_48
    del primals_123
    buf193 = reinterpret_tensor(buf165, (1024, 3072), (3072, 1), 0); del buf165  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf192, (1024, 768), (768, 1), 0), permute_167, out=buf193)
    del permute_167
    buf194 = reinterpret_tensor(buf184, (3072, 768), (768, 1), 0); del buf184  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_168, reinterpret_tensor(buf192, (1024, 768), (768, 1), 0), out=buf194)
    del permute_168
    buf195 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf196 = reinterpret_tensor(buf193, (2, 512, 3072), (1572864, 3072, 1), 0); del buf193  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_37(c_void_p(buf196.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(tanh_5.data_ptr()), c_void_p(buf195.data_ptr()))
    del addmm_22
    del tanh_5
    buf197 = buf185; del buf185  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf196, (1024, 3072), (3072, 1), 0), permute_169, out=buf197)
    del permute_169
    buf198 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_170, reinterpret_tensor(buf196, (1024, 3072), (3072, 1), 0), out=buf198)
    del permute_170
    buf199 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf200 = buf189; del buf189  # reuse
    buf201 = buf188; del buf188  # reuse
    buf202 = empty((768, ), device='cpu', dtype=torch.float32)
    buf203 = empty((768, ), device='cpu', dtype=torch.float32)
    buf204 = buf192; del buf192  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_38(c_void_p(buf204.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(mul_42.data_ptr()), c_void_p(div_43.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()))
    del div_43
    del mul_42
    del primals_121
    buf205 = buf197; del buf197  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf204, (1024, 768), (768, 1), 0), permute_171, out=buf205)
    del permute_171
    buf206 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_172, reinterpret_tensor(buf204, (1024, 768), (768, 1), 0), out=buf206)
    del permute_172
    buf207 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf208 = reinterpret_tensor(buf182, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf182  # reuse
    cpp_fused_clone_sum_39(c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()))
    buf209 = reinterpret_tensor(buf205, (24, 512, 64), (32768, 64, 1), 0); del buf205  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_174, reinterpret_tensor(buf208, (24, 512, 64), (32768, 64, 1), 0), out=buf209)
    del permute_174
    buf210 = reinterpret_tensor(buf181, (24, 512, 512), (262144, 512, 1), 0); del buf181  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf208, (24, 512, 64), (32768, 64, 1), 0), permute_175, out=buf210)
    del permute_175
    buf211 = buf180; del buf180  # reuse
    buf212 = reinterpret_tensor(buf210, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf210  # reuse
    cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_40(c_void_p(buf212.data_ptr()), c_void_p(alias_37.data_ptr()), c_void_p(slice_24.data_ptr()), c_void_p(buf211.data_ptr()))
    del alias_37
    del slice_24
    buf213 = reinterpret_tensor(buf208, (24, 64, 512), (32768, 512, 1), 0); del buf208  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_176, reinterpret_tensor(buf212, (24, 512, 512), (262144, 512, 1), 0), out=buf213)
    del permute_176
    buf214 = buf178; del buf178  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf212, (24, 512, 512), (262144, 512, 1), 0), permute_177, out=buf214)
    del permute_177
    buf215 = empty((2, 512, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_41(c_void_p(buf214.data_ptr()), c_void_p(tangents_12.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(tangents_13.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf215.data_ptr()))
    del tangents_12
    del tangents_13
    buf216 = reinterpret_tensor(buf214, (1024, 768), (768, 1), 0); del buf214  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf215, (1024, 2304), (2304, 1), 0), permute_182, out=buf216)
    del permute_182
    buf217 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_183, reinterpret_tensor(buf215, (1024, 2304), (2304, 1), 0), out=buf217)
    del permute_183
    buf218 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf219 = buf201; del buf201  # reuse
    buf220 = buf200; del buf200  # reuse
    buf221 = empty((768, ), device='cpu', dtype=torch.float32)
    buf222 = empty((768, ), device='cpu', dtype=torch.float32)
    buf223 = buf204; del buf204  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_42(c_void_p(buf223.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(mul_40.data_ptr()), c_void_p(div_45.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()))
    del div_45
    del mul_40
    del primals_119
    buf224 = reinterpret_tensor(buf196, (1024, 3072), (3072, 1), 0); del buf196  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf223, (1024, 768), (768, 1), 0), permute_184, out=buf224)
    del permute_184
    buf225 = reinterpret_tensor(buf215, (3072, 768), (768, 1), 0); del buf215  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_185, reinterpret_tensor(buf223, (1024, 768), (768, 1), 0), out=buf225)
    del permute_185
    buf226 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf227 = reinterpret_tensor(buf224, (2, 512, 3072), (1572864, 3072, 1), 0); del buf224  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_43(c_void_p(buf227.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(addmm_18.data_ptr()), c_void_p(tanh_4.data_ptr()), c_void_p(buf226.data_ptr()))
    del addmm_18
    del tanh_4
    buf228 = buf216; del buf216  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf227, (1024, 3072), (3072, 1), 0), permute_186, out=buf228)
    del permute_186
    buf229 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_187, reinterpret_tensor(buf227, (1024, 3072), (3072, 1), 0), out=buf229)
    del permute_187
    buf230 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf231 = buf220; del buf220  # reuse
    buf232 = buf219; del buf219  # reuse
    buf233 = empty((768, ), device='cpu', dtype=torch.float32)
    buf234 = empty((768, ), device='cpu', dtype=torch.float32)
    buf235 = buf223; del buf223  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_44(c_void_p(buf235.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(mul_34.data_ptr()), c_void_p(div_46.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()))
    del div_46
    del mul_34
    del primals_117
    buf236 = buf228; del buf228  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf235, (1024, 768), (768, 1), 0), permute_188, out=buf236)
    del permute_188
    buf237 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_189, reinterpret_tensor(buf235, (1024, 768), (768, 1), 0), out=buf237)
    del permute_189
    buf238 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf239 = reinterpret_tensor(buf213, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf213  # reuse
    cpp_fused_clone_sum_45(c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()))
    buf240 = reinterpret_tensor(buf236, (24, 512, 64), (32768, 64, 1), 0); del buf236  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_191, reinterpret_tensor(buf239, (24, 512, 64), (32768, 64, 1), 0), out=buf240)
    del permute_191
    buf241 = reinterpret_tensor(buf212, (24, 512, 512), (262144, 512, 1), 0); del buf212  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf239, (24, 512, 64), (32768, 64, 1), 0), permute_192, out=buf241)
    del permute_192
    buf242 = buf211; del buf211  # reuse
    buf243 = reinterpret_tensor(buf241, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf241  # reuse
    cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_46(c_void_p(buf243.data_ptr()), c_void_p(alias_39.data_ptr()), c_void_p(slice_20.data_ptr()), c_void_p(buf242.data_ptr()))
    del alias_39
    del slice_20
    buf244 = reinterpret_tensor(buf239, (24, 64, 512), (32768, 512, 1), 0); del buf239  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_193, reinterpret_tensor(buf243, (24, 512, 512), (262144, 512, 1), 0), out=buf244)
    del permute_193
    buf245 = buf209; del buf209  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf243, (24, 512, 512), (262144, 512, 1), 0), permute_194, out=buf245)
    del permute_194
    buf246 = empty((2, 512, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_47(c_void_p(buf245.data_ptr()), c_void_p(tangents_10.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(tangents_11.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf246.data_ptr()))
    del tangents_10
    del tangents_11
    buf247 = reinterpret_tensor(buf245, (1024, 768), (768, 1), 0); del buf245  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf246, (1024, 2304), (2304, 1), 0), permute_199, out=buf247)
    del permute_199
    buf248 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_200, reinterpret_tensor(buf246, (1024, 2304), (2304, 1), 0), out=buf248)
    del permute_200
    buf249 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf250 = buf232; del buf232  # reuse
    buf251 = buf231; del buf231  # reuse
    buf252 = empty((768, ), device='cpu', dtype=torch.float32)
    buf253 = empty((768, ), device='cpu', dtype=torch.float32)
    buf254 = buf235; del buf235  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_48(c_void_p(buf254.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(mul_32.data_ptr()), c_void_p(div_48.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()))
    del div_48
    del mul_32
    del primals_115
    buf255 = reinterpret_tensor(buf227, (1024, 3072), (3072, 1), 0); del buf227  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf254, (1024, 768), (768, 1), 0), permute_201, out=buf255)
    del permute_201
    buf256 = reinterpret_tensor(buf246, (3072, 768), (768, 1), 0); del buf246  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_202, reinterpret_tensor(buf254, (1024, 768), (768, 1), 0), out=buf256)
    del permute_202
    buf257 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf258 = reinterpret_tensor(buf255, (2, 512, 3072), (1572864, 3072, 1), 0); del buf255  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_49(c_void_p(buf258.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(addmm_14.data_ptr()), c_void_p(tanh_3.data_ptr()), c_void_p(buf257.data_ptr()))
    del addmm_14
    del tanh_3
    buf259 = buf247; del buf247  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf258, (1024, 3072), (3072, 1), 0), permute_203, out=buf259)
    del permute_203
    buf260 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_204, reinterpret_tensor(buf258, (1024, 3072), (3072, 1), 0), out=buf260)
    del permute_204
    buf261 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf262 = buf251; del buf251  # reuse
    buf263 = buf250; del buf250  # reuse
    buf264 = empty((768, ), device='cpu', dtype=torch.float32)
    buf265 = empty((768, ), device='cpu', dtype=torch.float32)
    buf266 = buf254; del buf254  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_50(c_void_p(buf266.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(mul_26.data_ptr()), c_void_p(div_49.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()))
    del div_49
    del mul_26
    del primals_113
    buf267 = buf259; del buf259  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf266, (1024, 768), (768, 1), 0), permute_205, out=buf267)
    del permute_205
    buf268 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_206, reinterpret_tensor(buf266, (1024, 768), (768, 1), 0), out=buf268)
    del permute_206
    buf269 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf270 = reinterpret_tensor(buf244, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf244  # reuse
    cpp_fused_clone_sum_51(c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()))
    buf271 = reinterpret_tensor(buf267, (24, 512, 64), (32768, 64, 1), 0); del buf267  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_208, reinterpret_tensor(buf270, (24, 512, 64), (32768, 64, 1), 0), out=buf271)
    del permute_208
    buf272 = reinterpret_tensor(buf243, (24, 512, 512), (262144, 512, 1), 0); del buf243  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf270, (24, 512, 64), (32768, 64, 1), 0), permute_209, out=buf272)
    del permute_209
    buf273 = buf242; del buf242  # reuse
    buf274 = reinterpret_tensor(buf272, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf272  # reuse
    cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_52(c_void_p(buf274.data_ptr()), c_void_p(alias_41.data_ptr()), c_void_p(slice_16.data_ptr()), c_void_p(buf273.data_ptr()))
    del alias_41
    del slice_16
    buf275 = reinterpret_tensor(buf270, (24, 64, 512), (32768, 512, 1), 0); del buf270  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_210, reinterpret_tensor(buf274, (24, 512, 512), (262144, 512, 1), 0), out=buf275)
    del permute_210
    buf276 = buf240; del buf240  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf274, (24, 512, 512), (262144, 512, 1), 0), permute_211, out=buf276)
    del permute_211
    buf277 = empty((2, 512, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_53(c_void_p(buf276.data_ptr()), c_void_p(tangents_8.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(tangents_9.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf277.data_ptr()))
    del tangents_8
    del tangents_9
    buf278 = reinterpret_tensor(buf276, (1024, 768), (768, 1), 0); del buf276  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf277, (1024, 2304), (2304, 1), 0), permute_216, out=buf278)
    del permute_216
    buf279 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_217, reinterpret_tensor(buf277, (1024, 2304), (2304, 1), 0), out=buf279)
    del permute_217
    buf280 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf281 = buf263; del buf263  # reuse
    buf282 = buf262; del buf262  # reuse
    buf283 = empty((768, ), device='cpu', dtype=torch.float32)
    buf284 = empty((768, ), device='cpu', dtype=torch.float32)
    buf285 = buf266; del buf266  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_54(c_void_p(buf285.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(mul_24.data_ptr()), c_void_p(div_51.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()))
    del div_51
    del mul_24
    del primals_111
    buf286 = reinterpret_tensor(buf258, (1024, 3072), (3072, 1), 0); del buf258  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf285, (1024, 768), (768, 1), 0), permute_218, out=buf286)
    del permute_218
    buf287 = reinterpret_tensor(buf277, (3072, 768), (768, 1), 0); del buf277  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_219, reinterpret_tensor(buf285, (1024, 768), (768, 1), 0), out=buf287)
    del permute_219
    buf288 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf289 = reinterpret_tensor(buf286, (2, 512, 3072), (1572864, 3072, 1), 0); del buf286  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_55(c_void_p(buf289.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(tanh_2.data_ptr()), c_void_p(buf288.data_ptr()))
    del addmm_10
    del tanh_2
    buf290 = buf278; del buf278  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf289, (1024, 3072), (3072, 1), 0), permute_220, out=buf290)
    del permute_220
    buf291 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_221, reinterpret_tensor(buf289, (1024, 3072), (3072, 1), 0), out=buf291)
    del permute_221
    buf292 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf293 = buf282; del buf282  # reuse
    buf294 = buf281; del buf281  # reuse
    buf295 = empty((768, ), device='cpu', dtype=torch.float32)
    buf296 = empty((768, ), device='cpu', dtype=torch.float32)
    buf297 = buf285; del buf285  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_56(c_void_p(buf297.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(mul_18.data_ptr()), c_void_p(div_52.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()))
    del div_52
    del mul_18
    del primals_109
    buf298 = buf290; del buf290  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (1024, 768), (768, 1), 0), permute_222, out=buf298)
    del permute_222
    buf299 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_223, reinterpret_tensor(buf297, (1024, 768), (768, 1), 0), out=buf299)
    del permute_223
    buf300 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf301 = reinterpret_tensor(buf275, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf275  # reuse
    cpp_fused_clone_sum_57(c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()))
    buf302 = reinterpret_tensor(buf298, (24, 512, 64), (32768, 64, 1), 0); del buf298  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_225, reinterpret_tensor(buf301, (24, 512, 64), (32768, 64, 1), 0), out=buf302)
    del permute_225
    buf303 = reinterpret_tensor(buf274, (24, 512, 512), (262144, 512, 1), 0); del buf274  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf301, (24, 512, 64), (32768, 64, 1), 0), permute_226, out=buf303)
    del permute_226
    buf304 = buf273; del buf273  # reuse
    buf305 = reinterpret_tensor(buf303, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf303  # reuse
    cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_58(c_void_p(buf305.data_ptr()), c_void_p(alias_43.data_ptr()), c_void_p(slice_12.data_ptr()), c_void_p(buf304.data_ptr()))
    del alias_43
    del slice_12
    buf306 = reinterpret_tensor(buf301, (24, 64, 512), (32768, 512, 1), 0); del buf301  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_227, reinterpret_tensor(buf305, (24, 512, 512), (262144, 512, 1), 0), out=buf306)
    del permute_227
    buf307 = buf271; del buf271  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf305, (24, 512, 512), (262144, 512, 1), 0), permute_228, out=buf307)
    del permute_228
    buf308 = empty((2, 512, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_59(c_void_p(buf307.data_ptr()), c_void_p(tangents_6.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(tangents_7.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf308.data_ptr()))
    del tangents_6
    del tangents_7
    buf309 = reinterpret_tensor(buf307, (1024, 768), (768, 1), 0); del buf307  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf308, (1024, 2304), (2304, 1), 0), permute_233, out=buf309)
    del permute_233
    buf310 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_234, reinterpret_tensor(buf308, (1024, 2304), (2304, 1), 0), out=buf310)
    del permute_234
    buf311 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf312 = buf294; del buf294  # reuse
    buf313 = buf293; del buf293  # reuse
    buf314 = empty((768, ), device='cpu', dtype=torch.float32)
    buf315 = empty((768, ), device='cpu', dtype=torch.float32)
    buf316 = buf297; del buf297  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_60(c_void_p(buf316.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(mul_16.data_ptr()), c_void_p(div_54.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()))
    del div_54
    del mul_16
    del primals_107
    buf317 = reinterpret_tensor(buf289, (1024, 3072), (3072, 1), 0); del buf289  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf316, (1024, 768), (768, 1), 0), permute_235, out=buf317)
    del permute_235
    buf318 = reinterpret_tensor(buf308, (3072, 768), (768, 1), 0); del buf308  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_236, reinterpret_tensor(buf316, (1024, 768), (768, 1), 0), out=buf318)
    del permute_236
    buf319 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf320 = reinterpret_tensor(buf317, (2, 512, 3072), (1572864, 3072, 1), 0); del buf317  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_61(c_void_p(buf320.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(addmm_6.data_ptr()), c_void_p(tanh_1.data_ptr()), c_void_p(buf319.data_ptr()))
    del addmm_6
    del tanh_1
    buf321 = buf309; del buf309  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf320, (1024, 3072), (3072, 1), 0), permute_237, out=buf321)
    del permute_237
    buf322 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_238, reinterpret_tensor(buf320, (1024, 3072), (3072, 1), 0), out=buf322)
    del permute_238
    buf323 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf324 = buf313; del buf313  # reuse
    buf325 = buf312; del buf312  # reuse
    buf326 = empty((768, ), device='cpu', dtype=torch.float32)
    buf327 = empty((768, ), device='cpu', dtype=torch.float32)
    buf328 = buf316; del buf316  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_62(c_void_p(buf328.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(mul_10.data_ptr()), c_void_p(div_55.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()))
    del div_55
    del mul_10
    del primals_105
    buf329 = buf321; del buf321  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf328, (1024, 768), (768, 1), 0), permute_239, out=buf329)
    del permute_239
    buf330 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_240, reinterpret_tensor(buf328, (1024, 768), (768, 1), 0), out=buf330)
    del permute_240
    buf331 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf332 = reinterpret_tensor(buf306, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf306  # reuse
    cpp_fused_clone_sum_63(c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()))
    buf333 = reinterpret_tensor(buf329, (24, 512, 64), (32768, 64, 1), 0); del buf329  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_242, reinterpret_tensor(buf332, (24, 512, 64), (32768, 64, 1), 0), out=buf333)
    del permute_242
    buf334 = reinterpret_tensor(buf305, (24, 512, 512), (262144, 512, 1), 0); del buf305  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf332, (24, 512, 64), (32768, 64, 1), 0), permute_243, out=buf334)
    del permute_243
    buf335 = buf304; del buf304  # reuse
    buf336 = reinterpret_tensor(buf334, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf334  # reuse
    cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_64(c_void_p(buf336.data_ptr()), c_void_p(alias_45.data_ptr()), c_void_p(slice_8.data_ptr()), c_void_p(buf335.data_ptr()))
    del alias_45
    del slice_8
    buf337 = reinterpret_tensor(buf332, (24, 64, 512), (32768, 512, 1), 0); del buf332  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_244, reinterpret_tensor(buf336, (24, 512, 512), (262144, 512, 1), 0), out=buf337)
    del permute_244
    buf338 = buf302; del buf302  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf336, (24, 512, 512), (262144, 512, 1), 0), permute_245, out=buf338)
    del permute_245
    buf339 = empty((2, 512, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_65(c_void_p(buf338.data_ptr()), c_void_p(tangents_4.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(tangents_5.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf339.data_ptr()))
    del tangents_4
    del tangents_5
    buf340 = reinterpret_tensor(buf338, (1024, 768), (768, 1), 0); del buf338  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf339, (1024, 2304), (2304, 1), 0), permute_250, out=buf340)
    del permute_250
    buf341 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_251, reinterpret_tensor(buf339, (1024, 2304), (2304, 1), 0), out=buf341)
    del permute_251
    buf342 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf343 = buf325; del buf325  # reuse
    buf344 = buf324; del buf324  # reuse
    buf345 = empty((768, ), device='cpu', dtype=torch.float32)
    buf346 = empty((768, ), device='cpu', dtype=torch.float32)
    buf347 = buf328; del buf328  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_66(c_void_p(buf347.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(mul_8.data_ptr()), c_void_p(div_57.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()))
    del div_57
    del mul_8
    del primals_103
    buf348 = reinterpret_tensor(buf320, (1024, 3072), (3072, 1), 0); del buf320  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf347, (1024, 768), (768, 1), 0), permute_252, out=buf348)
    del permute_252
    buf349 = reinterpret_tensor(buf339, (3072, 768), (768, 1), 0); del buf339  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_253, reinterpret_tensor(buf347, (1024, 768), (768, 1), 0), out=buf349)
    del permute_253
    buf350 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf351 = reinterpret_tensor(buf348, (2, 512, 3072), (1572864, 3072, 1), 0); del buf348  # reuse
    cpp_fused_add_mul_pow_sum_tanh_backward_67(c_void_p(buf351.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(addmm_2.data_ptr()), c_void_p(tanh.data_ptr()), c_void_p(buf350.data_ptr()))
    del addmm_2
    del tanh
    buf352 = buf340; del buf340  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf351, (1024, 3072), (3072, 1), 0), permute_254, out=buf352)
    del permute_254
    buf353 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_255, reinterpret_tensor(buf351, (1024, 3072), (3072, 1), 0), out=buf353)
    del permute_255
    buf354 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf355 = buf344; del buf344  # reuse
    buf356 = buf343; del buf343  # reuse
    buf357 = empty((768, ), device='cpu', dtype=torch.float32)
    buf358 = empty((768, ), device='cpu', dtype=torch.float32)
    buf359 = buf347; del buf347  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_68(c_void_p(buf359.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(mul_2.data_ptr()), c_void_p(div_58.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()))
    del buf351
    del div_58
    del mul_2
    del primals_101
    buf360 = buf352; del buf352  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf359, (1024, 768), (768, 1), 0), permute_256, out=buf360)
    del permute_256
    buf361 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_257, reinterpret_tensor(buf359, (1024, 768), (768, 1), 0), out=buf361)
    del permute_257
    buf362 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf363 = reinterpret_tensor(buf337, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf337  # reuse
    cpp_fused_clone_sum_69(c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()))
    buf364 = reinterpret_tensor(buf360, (24, 512, 64), (32768, 64, 1), 0); del buf360  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_259, reinterpret_tensor(buf363, (24, 512, 64), (32768, 64, 1), 0), out=buf364)
    del permute_259
    buf365 = reinterpret_tensor(buf336, (24, 512, 512), (262144, 512, 1), 0); del buf336  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf363, (24, 512, 64), (32768, 64, 1), 0), permute_260, out=buf365)
    del permute_260
    buf366 = buf335; del buf335  # reuse
    buf367 = reinterpret_tensor(buf365, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf365  # reuse
    cpp_fused__softmax_backward_data_div_full_scalar_tensor_where_70(c_void_p(buf367.data_ptr()), c_void_p(alias_47.data_ptr()), c_void_p(slice_4.data_ptr()), c_void_p(buf366.data_ptr()))
    del alias_47
    del buf366
    del slice_4
    buf368 = reinterpret_tensor(buf363, (24, 64, 512), (32768, 512, 1), 0); del buf363  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_261, reinterpret_tensor(buf367, (24, 512, 512), (262144, 512, 1), 0), out=buf368)
    del permute_261
    buf369 = buf333; del buf333  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf367, (24, 512, 512), (262144, 512, 1), 0), permute_262, out=buf369)
    del buf367
    del permute_262
    buf370 = empty((2, 512, 2304), device='cpu', dtype=torch.float32)
    cpp_fused_cat_71(c_void_p(buf369.data_ptr()), c_void_p(tangents_2.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(tangents_3.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf370.data_ptr()))
    del tangents_2
    del tangents_3
    buf371 = reinterpret_tensor(buf369, (1024, 768), (768, 1), 0); del buf369  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf370, (1024, 2304), (2304, 1), 0), permute_267, out=buf371)
    del permute_267
    buf372 = empty((768, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(permute_268, reinterpret_tensor(buf370, (1024, 2304), (2304, 1), 0), out=buf372)
    del permute_268
    buf373 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf374 = buf356; del buf356  # reuse
    buf375 = buf355; del buf355  # reuse
    buf376 = empty((768, ), device='cpu', dtype=torch.float32)
    buf377 = empty((768, ), device='cpu', dtype=torch.float32)
    buf378 = buf359; del buf359  # reuse
    buf384 = reinterpret_tensor(buf368, (2, 512, 768), (393216, 768, 1), 0); del buf368  # reuse
    buf379 = reinterpret_tensor(buf364, (1024, 768), (768, 1), 0); del buf364  # reuse
    buf380 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_dense_backward_native_layer_norm_backward_scalar_tensor_sum_72(c_void_p(buf378.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(mul.data_ptr()), c_void_p(div_60.data_ptr()), c_void_p(view.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()))
    del buf370
    del buf371
    del buf374
    del buf375
    del buf378
    del div_60
    del mul
    del primals_99
    aten.index_put_(buf379, [view_1], buf380, True)
    del buf380
    del view_1
    buf383 = empty((50257, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_73(c_void_p(buf383.data_ptr()))
    aten.index_put_(buf383, [view], buf384, True)
    del buf384
    del view
    return (reinterpret_tensor(buf373, (2304, ), (1, ), 0), buf372, reinterpret_tensor(buf362, (768, ), (1, ), 0), buf361, reinterpret_tensor(buf354, (3072, ), (1, ), 0), buf353, reinterpret_tensor(buf350, (768, ), (1, ), 0), buf349, reinterpret_tensor(buf342, (2304, ), (1, ), 0), buf341, reinterpret_tensor(buf331, (768, ), (1, ), 0), buf330, reinterpret_tensor(buf323, (3072, ), (1, ), 0), buf322, reinterpret_tensor(buf319, (768, ), (1, ), 0), buf318, reinterpret_tensor(buf311, (2304, ), (1, ), 0), buf310, reinterpret_tensor(buf300, (768, ), (1, ), 0), buf299, reinterpret_tensor(buf292, (3072, ), (1, ), 0), buf291, reinterpret_tensor(buf288, (768, ), (1, ), 0), buf287, reinterpret_tensor(buf280, (2304, ), (1, ), 0), buf279, reinterpret_tensor(buf269, (768, ), (1, ), 0), buf268, reinterpret_tensor(buf261, (3072, ), (1, ), 0), buf260, reinterpret_tensor(buf257, (768, ), (1, ), 0), buf256, reinterpret_tensor(buf249, (2304, ), (1, ), 0), buf248, reinterpret_tensor(buf238, (768, ), (1, ), 0), buf237, reinterpret_tensor(buf230, (3072, ), (1, ), 0), buf229, reinterpret_tensor(buf226, (768, ), (1, ), 0), buf225, reinterpret_tensor(buf218, (2304, ), (1, ), 0), buf217, reinterpret_tensor(buf207, (768, ), (1, ), 0), buf206, reinterpret_tensor(buf199, (3072, ), (1, ), 0), buf198, reinterpret_tensor(buf195, (768, ), (1, ), 0), buf194, reinterpret_tensor(buf187, (2304, ), (1, ), 0), buf186, reinterpret_tensor(buf176, (768, ), (1, ), 0), buf175, reinterpret_tensor(buf168, (3072, ), (1, ), 0), buf167, reinterpret_tensor(buf164, (768, ), (1, ), 0), buf163, reinterpret_tensor(buf156, (2304, ), (1, ), 0), buf155, reinterpret_tensor(buf145, (768, ), (1, ), 0), buf144, reinterpret_tensor(buf137, (3072, ), (1, ), 0), buf136, reinterpret_tensor(buf133, (768, ), (1, ), 0), buf132, reinterpret_tensor(buf125, (2304, ), (1, ), 0), buf124, reinterpret_tensor(buf114, (768, ), (1, ), 0), buf113, reinterpret_tensor(buf106, (3072, ), (1, ), 0), buf105, reinterpret_tensor(buf102, (768, ), (1, ), 0), buf101, reinterpret_tensor(buf94, (2304, ), (1, ), 0), buf93, reinterpret_tensor(buf83, (768, ), (1, ), 0), buf82, reinterpret_tensor(buf75, (3072, ), (1, ), 0), buf74, reinterpret_tensor(buf71, (768, ), (1, ), 0), buf70, reinterpret_tensor(buf63, (2304, ), (1, ), 0), buf62, reinterpret_tensor(buf52, (768, ), (1, ), 0), buf51, reinterpret_tensor(buf44, (3072, ), (1, ), 0), buf43, reinterpret_tensor(buf40, (768, ), (1, ), 0), buf39, reinterpret_tensor(buf32, (2304, ), (1, ), 0), buf31, reinterpret_tensor(buf21, (768, ), (1, ), 0), buf20, reinterpret_tensor(buf13, (3072, ), (1, ), 0), buf12, reinterpret_tensor(buf9, (768, ), (1, ), 0), buf8, buf383, buf379, buf376, buf377, buf357, buf358, buf345, buf346, buf326, buf327, buf314, buf315, buf295, buf296, buf283, buf284, buf264, buf265, buf252, buf253, buf233, buf234, buf221, buf222, buf202, buf203, buf190, buf191, buf171, buf172, buf159, buf160, buf140, buf141, buf128, buf129, buf109, buf110, buf97, buf98, buf78, buf79, buf66, buf67, buf47, buf48, buf35, buf36, buf16, buf17, buf5, buf6, reinterpret_tensor(buf0, (50257, 768), (768, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, )


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
    view = rand_strided((2, 512), (512, 1), device='cpu', dtype=torch.int64)
    view_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    mul = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    slice_4 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    mul_2 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    addmm_2 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    mul_8 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    slice_8 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    mul_10 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    addmm_6 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_1 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    mul_16 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    slice_12 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    mul_18 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_2 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    mul_24 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    slice_16 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    mul_26 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    addmm_14 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_3 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    mul_32 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    slice_20 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    mul_34 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    addmm_18 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_4 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    mul_40 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    slice_24 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    mul_42 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_5 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    mul_48 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    slice_28 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    mul_50 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    addmm_26 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_6 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    mul_56 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    slice_32 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    mul_58 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    addmm_30 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_7 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    mul_64 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    slice_36 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    mul_66 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_8 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    mul_72 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    slice_40 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    mul_74 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    addmm_38 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_9 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    mul_80 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    slice_44 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    mul_82 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    addmm_42 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_10 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    mul_88 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    slice_48 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    mul_90 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    addmm_46 = rand_strided((1024, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    tanh_11 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cpu', dtype=torch.float32)
    mul_96 = rand_strided((2, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_219 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_63 = rand_strided((50257, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_65 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_66 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_67 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_68 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_25 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_69 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_70 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_72 = rand_strided((24, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_73 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_25 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_74 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_75 = rand_strided((24, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_80 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_81 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_27 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_82 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_83 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_84 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_85 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_28 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_86 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_87 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_89 = rand_strided((24, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_90 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_27 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_91 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_92 = rand_strided((24, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_97 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_98 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_30 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_99 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_100 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_101 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_102 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_31 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_103 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_104 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_106 = rand_strided((24, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_107 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_29 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_108 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_109 = rand_strided((24, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_114 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_115 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_33 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_116 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_117 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_118 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_119 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_34 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_120 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_121 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_123 = rand_strided((24, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_124 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_31 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_125 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_126 = rand_strided((24, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_131 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_132 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_36 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_133 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_134 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_135 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_136 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_37 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_137 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_138 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_140 = rand_strided((24, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_141 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_33 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_142 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_143 = rand_strided((24, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_148 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_149 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_39 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_150 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_151 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_152 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_153 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_40 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_154 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_155 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_157 = rand_strided((24, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_158 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_35 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_159 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_160 = rand_strided((24, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_165 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_166 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_42 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_167 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_168 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_169 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_170 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_43 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_171 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_172 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_174 = rand_strided((24, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_175 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_37 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_176 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_177 = rand_strided((24, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_182 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_183 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_45 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_184 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_185 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_186 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_187 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_46 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_188 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_189 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_191 = rand_strided((24, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_192 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_39 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_193 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_194 = rand_strided((24, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_199 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_200 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_48 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_201 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_202 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_203 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_204 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_49 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_205 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_206 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_208 = rand_strided((24, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_209 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_41 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_210 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_211 = rand_strided((24, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_216 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_217 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_51 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_218 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_219 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_220 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_221 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_52 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_222 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_223 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_225 = rand_strided((24, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_226 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_43 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_227 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_228 = rand_strided((24, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_233 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_234 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_54 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_235 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_236 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_237 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_238 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_55 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_239 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_240 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_242 = rand_strided((24, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_243 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_45 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_244 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_245 = rand_strided((24, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_250 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_251 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_57 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_252 = rand_strided((768, 3072), (1, 768), device='cpu', dtype=torch.float32)
    permute_253 = rand_strided((3072, 1024), (1, 3072), device='cpu', dtype=torch.float32)
    permute_254 = rand_strided((3072, 768), (1, 3072), device='cpu', dtype=torch.float32)
    permute_255 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_58 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_256 = rand_strided((768, 768), (1, 768), device='cpu', dtype=torch.float32)
    permute_257 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    permute_259 = rand_strided((24, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_260 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_47 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_261 = rand_strided((24, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_262 = rand_strided((24, 512, 64), (32768, 1, 512), device='cpu', dtype=torch.float32)
    permute_267 = rand_strided((2304, 768), (1, 2304), device='cpu', dtype=torch.float32)
    permute_268 = rand_strided((768, 1024), (1, 768), device='cpu', dtype=torch.float32)
    div_60 = rand_strided((2, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((2, 512, 50257), (25731584, 50257, 1), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_3 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_4 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_5 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_6 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_7 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_8 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_9 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_10 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_11 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_12 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_13 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_14 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_15 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_16 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_17 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_18 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_19 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_20 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_21 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_22 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_23 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_24 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    tangents_25 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, view, view_1, mul, slice_4, mul_2, addmm_2, tanh, mul_8, slice_8, mul_10, addmm_6, tanh_1, mul_16, slice_12, mul_18, addmm_10, tanh_2, mul_24, slice_16, mul_26, addmm_14, tanh_3, mul_32, slice_20, mul_34, addmm_18, tanh_4, mul_40, slice_24, mul_42, addmm_22, tanh_5, mul_48, slice_28, mul_50, addmm_26, tanh_6, mul_56, slice_32, mul_58, addmm_30, tanh_7, mul_64, slice_36, mul_66, addmm_34, tanh_8, mul_72, slice_40, mul_74, addmm_38, tanh_9, mul_80, slice_44, mul_82, addmm_42, tanh_10, mul_88, slice_48, mul_90, addmm_46, tanh_11, mul_96, view_219, permute_63, div_24, permute_65, permute_66, permute_67, permute_68, div_25, permute_69, permute_70, permute_72, permute_73, alias_25, permute_74, permute_75, permute_80, permute_81, div_27, permute_82, permute_83, permute_84, permute_85, div_28, permute_86, permute_87, permute_89, permute_90, alias_27, permute_91, permute_92, permute_97, permute_98, div_30, permute_99, permute_100, permute_101, permute_102, div_31, permute_103, permute_104, permute_106, permute_107, alias_29, permute_108, permute_109, permute_114, permute_115, div_33, permute_116, permute_117, permute_118, permute_119, div_34, permute_120, permute_121, permute_123, permute_124, alias_31, permute_125, permute_126, permute_131, permute_132, div_36, permute_133, permute_134, permute_135, permute_136, div_37, permute_137, permute_138, permute_140, permute_141, alias_33, permute_142, permute_143, permute_148, permute_149, div_39, permute_150, permute_151, permute_152, permute_153, div_40, permute_154, permute_155, permute_157, permute_158, alias_35, permute_159, permute_160, permute_165, permute_166, div_42, permute_167, permute_168, permute_169, permute_170, div_43, permute_171, permute_172, permute_174, permute_175, alias_37, permute_176, permute_177, permute_182, permute_183, div_45, permute_184, permute_185, permute_186, permute_187, div_46, permute_188, permute_189, permute_191, permute_192, alias_39, permute_193, permute_194, permute_199, permute_200, div_48, permute_201, permute_202, permute_203, permute_204, div_49, permute_205, permute_206, permute_208, permute_209, alias_41, permute_210, permute_211, permute_216, permute_217, div_51, permute_218, permute_219, permute_220, permute_221, div_52, permute_222, permute_223, permute_225, permute_226, alias_43, permute_227, permute_228, permute_233, permute_234, div_54, permute_235, permute_236, permute_237, permute_238, div_55, permute_239, permute_240, permute_242, permute_243, alias_45, permute_244, permute_245, permute_250, permute_251, div_57, permute_252, permute_253, permute_254, permute_255, div_58, permute_256, permute_257, permute_259, permute_260, alias_47, permute_261, permute_262, permute_267, permute_268, div_60, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_GPT2', benchmark_compiled_module)
