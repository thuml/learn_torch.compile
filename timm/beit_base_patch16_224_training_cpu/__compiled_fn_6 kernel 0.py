
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


cpp_fused_div_mul_native_layer_norm_backward_slice_backward_sum_0 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr4[static_cast<long>(x0)];
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (768L*x2) + (151296L*x1)));
                            auto tmp0 = c10::convert<int>(x2);
                            auto tmp1 = static_cast<int>(1);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = [&]
                            {
                                auto tmp4 = masked_load(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)), to_float_mask(tmp2));
                                auto tmp5 = static_cast<float>(196.0);
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 / tmp6;
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = to_float_mask(tmp2);
                            auto tmp11 = at::vec::Vectorized<float>(tmp9);
                            auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                            auto tmp14 = tmp12 * tmp13;
                            tmp_acc0_vec = tmp_acc0_vec + tmp14;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (768L*x0)), to_float_mask(tmp2));
                            auto tmp5 = static_cast<float>(196.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp9 = static_cast<float>(0.0);
                        auto tmp10 = to_float_mask(tmp2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp9);
                        auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                        auto tmp14 = tmp12 * tmp13;
                        tmp14.store(out_ptr6 + static_cast<long>(x2 + (768L*x1) + (151296L*x0)));
                    }
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mul_native_layer_norm_backward_slice_backward_sum_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp13 = in_ptr5[static_cast<long>(x1 + (197L*x0))];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (151296L*x0)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp20 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1) + (151296L*x0)));
                        auto tmp24 = out_ptr2[static_cast<long>(x1 + (197L*x0))];
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_ptr4 + static_cast<long>(x2 + (768L*x0)), to_float_mask(tmp2));
                            auto tmp5 = static_cast<float>(196.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp9 = static_cast<float>(0.0);
                        auto tmp10 = to_float_mask(tmp2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp9);
                        auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                        auto tmp16 = tmp14 * tmp15;
                        auto tmp17 = static_cast<float>(768.0);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp25 = at::vec::Vectorized<float>(tmp24);
                        auto tmp26 = tmp23 * tmp25;
                        auto tmp27 = tmp22 - tmp26;
                        auto tmp28 = at::vec::Vectorized<float>(tmp13);
                        auto tmp29 = tmp28 * tmp27;
                        auto tmp30 = tmp12 + tmp29;
                        tmp30.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (151296L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr6 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (2364L*x1) + (465708L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (38809L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (12L*x2) + (2364L*x1) + (465708L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (197L*x1) + (38809L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp6.store(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp5;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_put_new_zeros_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8784L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_clone_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(8L))
                            {
                                float tmp16[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), static_cast<long>(197L), tmp16, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                                    auto tmp1 = static_cast<int>(0);
                                    auto tmp2 = tmp0 >= tmp1;
                                    auto tmp3 = static_cast<int>(8);
                                    auto tmp4 = tmp0 < tmp3;
                                    auto tmp5 = [&]
                                    {
                                        auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                        auto tmp7 = static_cast<float>(0.3535533905932738);
                                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                        auto tmp9 = tmp6 * tmp8;
                                        return tmp9;
                                    }
                                    ;
                                    auto tmp10 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                                    auto tmp11 = tmp0 >= tmp3;
                                    auto tmp12 = static_cast<int>(16);
                                    auto tmp13 = tmp0 < tmp12;
                                    auto tmp14 = tmp11 & tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp17 = at::vec::Vectorized<float>::loadu(tmp16 + static_cast<long>(8L*x3_inner));
                                        auto tmp18 = static_cast<float>(0.3535533905932738);
                                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                                        auto tmp20 = tmp17 * tmp19;
                                        return tmp20;
                                    }
                                    ;
                                    auto tmp21 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp14));
                                    auto tmp22 = tmp0 >= tmp12;
                                    auto tmp23 = static_cast<int>(24);
                                    auto tmp24 = tmp0 < tmp23;
                                    auto tmp25 = [&]
                                    {
                                        auto tmp26 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp22));
                                        return tmp26;
                                    }
                                    ;
                                    auto tmp27 = decltype(tmp25())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp25(), to_float_mask(tmp22));
                                    auto tmp28 = to_float_mask(tmp14);
                                    auto tmp29 = decltype(tmp21)::blendv(tmp27, tmp21, tmp28);
                                    auto tmp30 = to_float_mask(tmp4);
                                    auto tmp31 = decltype(tmp10)::blendv(tmp29, tmp10, tmp30);
                                    tmp31.store(out_ptr0 + static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (2304L*x3_inner) + (453888L*x1)));
                                }
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = c10::convert<long>(x1 + (8L*x0));
                                auto tmp1 = static_cast<long>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<long>(8);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = [&]
                                {
                                    auto tmp6 = in_ptr0[static_cast<long>(x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp7 = static_cast<float>(0.3535533905932738);
                                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                    return tmp8;
                                }
                                ;
                                auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                                auto tmp10 = tmp0 >= tmp3;
                                auto tmp11 = static_cast<long>(16);
                                auto tmp12 = tmp0 < tmp11;
                                auto tmp13 = tmp10 & tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr1[static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp16 = static_cast<float>(0.3535533905932738);
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                auto tmp19 = tmp0 >= tmp11;
                                auto tmp20 = static_cast<long>(24);
                                auto tmp21 = tmp0 < tmp20;
                                auto tmp22 = [&]
                                {
                                    auto tmp23 = in_ptr2[static_cast<long>((-2420736L) + x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    return tmp23;
                                }
                                ;
                                auto tmp24 = tmp19 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                                auto tmp25 = tmp13 ? tmp18 : tmp24;
                                auto tmp26 = tmp4 ? tmp9 : tmp25;
                                out_ptr0[static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (453888L*x1))] = tmp26;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_backward_sum_7 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_8 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_mul_native_layer_norm_backward_sum_9 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_10 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (2364L*x1) + (465708L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (38809L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (12L*x2) + (2364L*x1) + (465708L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (197L*x1) + (38809L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp6.store(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp5;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_put_new_zeros_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8784L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_clone_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(8L))
                            {
                                float tmp16[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), static_cast<long>(197L), tmp16, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                                    auto tmp1 = static_cast<int>(0);
                                    auto tmp2 = tmp0 >= tmp1;
                                    auto tmp3 = static_cast<int>(8);
                                    auto tmp4 = tmp0 < tmp3;
                                    auto tmp5 = [&]
                                    {
                                        auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                        auto tmp7 = static_cast<float>(0.3535533905932738);
                                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                        auto tmp9 = tmp6 * tmp8;
                                        return tmp9;
                                    }
                                    ;
                                    auto tmp10 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                                    auto tmp11 = tmp0 >= tmp3;
                                    auto tmp12 = static_cast<int>(16);
                                    auto tmp13 = tmp0 < tmp12;
                                    auto tmp14 = tmp11 & tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp17 = at::vec::Vectorized<float>::loadu(tmp16 + static_cast<long>(8L*x3_inner));
                                        auto tmp18 = static_cast<float>(0.3535533905932738);
                                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                                        auto tmp20 = tmp17 * tmp19;
                                        return tmp20;
                                    }
                                    ;
                                    auto tmp21 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp14));
                                    auto tmp22 = tmp0 >= tmp12;
                                    auto tmp23 = static_cast<int>(24);
                                    auto tmp24 = tmp0 < tmp23;
                                    auto tmp25 = [&]
                                    {
                                        auto tmp26 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp22));
                                        return tmp26;
                                    }
                                    ;
                                    auto tmp27 = decltype(tmp25())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp25(), to_float_mask(tmp22));
                                    auto tmp28 = to_float_mask(tmp14);
                                    auto tmp29 = decltype(tmp21)::blendv(tmp27, tmp21, tmp28);
                                    auto tmp30 = to_float_mask(tmp4);
                                    auto tmp31 = decltype(tmp10)::blendv(tmp29, tmp10, tmp30);
                                    tmp31.store(out_ptr0 + static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (2304L*x3_inner) + (453888L*x1)));
                                }
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = c10::convert<long>(x1 + (8L*x0));
                                auto tmp1 = static_cast<long>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<long>(8);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = [&]
                                {
                                    auto tmp6 = in_ptr0[static_cast<long>(x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp7 = static_cast<float>(0.3535533905932738);
                                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                    return tmp8;
                                }
                                ;
                                auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                                auto tmp10 = tmp0 >= tmp3;
                                auto tmp11 = static_cast<long>(16);
                                auto tmp12 = tmp0 < tmp11;
                                auto tmp13 = tmp10 & tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr1[static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp16 = static_cast<float>(0.3535533905932738);
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                auto tmp19 = tmp0 >= tmp11;
                                auto tmp20 = static_cast<long>(24);
                                auto tmp21 = tmp0 < tmp20;
                                auto tmp22 = [&]
                                {
                                    auto tmp23 = in_ptr2[static_cast<long>((-2420736L) + x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    return tmp23;
                                }
                                ;
                                auto tmp24 = tmp19 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                                auto tmp25 = tmp13 ? tmp18 : tmp24;
                                auto tmp26 = tmp4 ? tmp9 : tmp25;
                                out_ptr0[static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (453888L*x1))] = tmp26;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_backward_sum_14 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_15 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_mul_native_layer_norm_backward_sum_16 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_17 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (2364L*x1) + (465708L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (38809L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (12L*x2) + (2364L*x1) + (465708L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (197L*x1) + (38809L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp6.store(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp5;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_put_new_zeros_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8784L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_clone_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(8L))
                            {
                                float tmp16[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), static_cast<long>(197L), tmp16, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                                    auto tmp1 = static_cast<int>(0);
                                    auto tmp2 = tmp0 >= tmp1;
                                    auto tmp3 = static_cast<int>(8);
                                    auto tmp4 = tmp0 < tmp3;
                                    auto tmp5 = [&]
                                    {
                                        auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                        auto tmp7 = static_cast<float>(0.3535533905932738);
                                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                        auto tmp9 = tmp6 * tmp8;
                                        return tmp9;
                                    }
                                    ;
                                    auto tmp10 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                                    auto tmp11 = tmp0 >= tmp3;
                                    auto tmp12 = static_cast<int>(16);
                                    auto tmp13 = tmp0 < tmp12;
                                    auto tmp14 = tmp11 & tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp17 = at::vec::Vectorized<float>::loadu(tmp16 + static_cast<long>(8L*x3_inner));
                                        auto tmp18 = static_cast<float>(0.3535533905932738);
                                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                                        auto tmp20 = tmp17 * tmp19;
                                        return tmp20;
                                    }
                                    ;
                                    auto tmp21 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp14));
                                    auto tmp22 = tmp0 >= tmp12;
                                    auto tmp23 = static_cast<int>(24);
                                    auto tmp24 = tmp0 < tmp23;
                                    auto tmp25 = [&]
                                    {
                                        auto tmp26 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp22));
                                        return tmp26;
                                    }
                                    ;
                                    auto tmp27 = decltype(tmp25())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp25(), to_float_mask(tmp22));
                                    auto tmp28 = to_float_mask(tmp14);
                                    auto tmp29 = decltype(tmp21)::blendv(tmp27, tmp21, tmp28);
                                    auto tmp30 = to_float_mask(tmp4);
                                    auto tmp31 = decltype(tmp10)::blendv(tmp29, tmp10, tmp30);
                                    tmp31.store(out_ptr0 + static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (2304L*x3_inner) + (453888L*x1)));
                                }
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = c10::convert<long>(x1 + (8L*x0));
                                auto tmp1 = static_cast<long>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<long>(8);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = [&]
                                {
                                    auto tmp6 = in_ptr0[static_cast<long>(x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp7 = static_cast<float>(0.3535533905932738);
                                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                    return tmp8;
                                }
                                ;
                                auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                                auto tmp10 = tmp0 >= tmp3;
                                auto tmp11 = static_cast<long>(16);
                                auto tmp12 = tmp0 < tmp11;
                                auto tmp13 = tmp10 & tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr1[static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp16 = static_cast<float>(0.3535533905932738);
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                auto tmp19 = tmp0 >= tmp11;
                                auto tmp20 = static_cast<long>(24);
                                auto tmp21 = tmp0 < tmp20;
                                auto tmp22 = [&]
                                {
                                    auto tmp23 = in_ptr2[static_cast<long>((-2420736L) + x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    return tmp23;
                                }
                                ;
                                auto tmp24 = tmp19 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                                auto tmp25 = tmp13 ? tmp18 : tmp24;
                                auto tmp26 = tmp4 ? tmp9 : tmp25;
                                out_ptr0[static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (453888L*x1))] = tmp26;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_backward_sum_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto in_ptr1 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_22 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_mul_native_layer_norm_backward_sum_23 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_24 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (2364L*x1) + (465708L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (38809L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (12L*x2) + (2364L*x1) + (465708L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (197L*x1) + (38809L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp6.store(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp5;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_put_new_zeros_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8784L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_clone_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(8L))
                            {
                                float tmp16[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), static_cast<long>(197L), tmp16, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                                    auto tmp1 = static_cast<int>(0);
                                    auto tmp2 = tmp0 >= tmp1;
                                    auto tmp3 = static_cast<int>(8);
                                    auto tmp4 = tmp0 < tmp3;
                                    auto tmp5 = [&]
                                    {
                                        auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                        auto tmp7 = static_cast<float>(0.3535533905932738);
                                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                        auto tmp9 = tmp6 * tmp8;
                                        return tmp9;
                                    }
                                    ;
                                    auto tmp10 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                                    auto tmp11 = tmp0 >= tmp3;
                                    auto tmp12 = static_cast<int>(16);
                                    auto tmp13 = tmp0 < tmp12;
                                    auto tmp14 = tmp11 & tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp17 = at::vec::Vectorized<float>::loadu(tmp16 + static_cast<long>(8L*x3_inner));
                                        auto tmp18 = static_cast<float>(0.3535533905932738);
                                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                                        auto tmp20 = tmp17 * tmp19;
                                        return tmp20;
                                    }
                                    ;
                                    auto tmp21 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp14));
                                    auto tmp22 = tmp0 >= tmp12;
                                    auto tmp23 = static_cast<int>(24);
                                    auto tmp24 = tmp0 < tmp23;
                                    auto tmp25 = [&]
                                    {
                                        auto tmp26 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp22));
                                        return tmp26;
                                    }
                                    ;
                                    auto tmp27 = decltype(tmp25())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp25(), to_float_mask(tmp22));
                                    auto tmp28 = to_float_mask(tmp14);
                                    auto tmp29 = decltype(tmp21)::blendv(tmp27, tmp21, tmp28);
                                    auto tmp30 = to_float_mask(tmp4);
                                    auto tmp31 = decltype(tmp10)::blendv(tmp29, tmp10, tmp30);
                                    tmp31.store(out_ptr0 + static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (2304L*x3_inner) + (453888L*x1)));
                                }
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = c10::convert<long>(x1 + (8L*x0));
                                auto tmp1 = static_cast<long>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<long>(8);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = [&]
                                {
                                    auto tmp6 = in_ptr0[static_cast<long>(x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp7 = static_cast<float>(0.3535533905932738);
                                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                    return tmp8;
                                }
                                ;
                                auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                                auto tmp10 = tmp0 >= tmp3;
                                auto tmp11 = static_cast<long>(16);
                                auto tmp12 = tmp0 < tmp11;
                                auto tmp13 = tmp10 & tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr1[static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp16 = static_cast<float>(0.3535533905932738);
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                auto tmp19 = tmp0 >= tmp11;
                                auto tmp20 = static_cast<long>(24);
                                auto tmp21 = tmp0 < tmp20;
                                auto tmp22 = [&]
                                {
                                    auto tmp23 = in_ptr2[static_cast<long>((-2420736L) + x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    return tmp23;
                                }
                                ;
                                auto tmp24 = tmp19 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                                auto tmp25 = tmp13 ? tmp18 : tmp24;
                                auto tmp26 = tmp4 ? tmp9 : tmp25;
                                out_ptr0[static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (453888L*x1))] = tmp26;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_backward_sum_28 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_29 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_mul_native_layer_norm_backward_sum_30 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_31 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (2364L*x1) + (465708L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (38809L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (12L*x2) + (2364L*x1) + (465708L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (197L*x1) + (38809L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp6.store(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp5;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_put_new_zeros_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8784L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_clone_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(8L))
                            {
                                float tmp16[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), static_cast<long>(197L), tmp16, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                                    auto tmp1 = static_cast<int>(0);
                                    auto tmp2 = tmp0 >= tmp1;
                                    auto tmp3 = static_cast<int>(8);
                                    auto tmp4 = tmp0 < tmp3;
                                    auto tmp5 = [&]
                                    {
                                        auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                        auto tmp7 = static_cast<float>(0.3535533905932738);
                                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                        auto tmp9 = tmp6 * tmp8;
                                        return tmp9;
                                    }
                                    ;
                                    auto tmp10 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                                    auto tmp11 = tmp0 >= tmp3;
                                    auto tmp12 = static_cast<int>(16);
                                    auto tmp13 = tmp0 < tmp12;
                                    auto tmp14 = tmp11 & tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp17 = at::vec::Vectorized<float>::loadu(tmp16 + static_cast<long>(8L*x3_inner));
                                        auto tmp18 = static_cast<float>(0.3535533905932738);
                                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                                        auto tmp20 = tmp17 * tmp19;
                                        return tmp20;
                                    }
                                    ;
                                    auto tmp21 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp14));
                                    auto tmp22 = tmp0 >= tmp12;
                                    auto tmp23 = static_cast<int>(24);
                                    auto tmp24 = tmp0 < tmp23;
                                    auto tmp25 = [&]
                                    {
                                        auto tmp26 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp22));
                                        return tmp26;
                                    }
                                    ;
                                    auto tmp27 = decltype(tmp25())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp25(), to_float_mask(tmp22));
                                    auto tmp28 = to_float_mask(tmp14);
                                    auto tmp29 = decltype(tmp21)::blendv(tmp27, tmp21, tmp28);
                                    auto tmp30 = to_float_mask(tmp4);
                                    auto tmp31 = decltype(tmp10)::blendv(tmp29, tmp10, tmp30);
                                    tmp31.store(out_ptr0 + static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (2304L*x3_inner) + (453888L*x1)));
                                }
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = c10::convert<long>(x1 + (8L*x0));
                                auto tmp1 = static_cast<long>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<long>(8);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = [&]
                                {
                                    auto tmp6 = in_ptr0[static_cast<long>(x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp7 = static_cast<float>(0.3535533905932738);
                                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                    return tmp8;
                                }
                                ;
                                auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                                auto tmp10 = tmp0 >= tmp3;
                                auto tmp11 = static_cast<long>(16);
                                auto tmp12 = tmp0 < tmp11;
                                auto tmp13 = tmp10 & tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr1[static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp16 = static_cast<float>(0.3535533905932738);
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                auto tmp19 = tmp0 >= tmp11;
                                auto tmp20 = static_cast<long>(24);
                                auto tmp21 = tmp0 < tmp20;
                                auto tmp22 = [&]
                                {
                                    auto tmp23 = in_ptr2[static_cast<long>((-2420736L) + x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    return tmp23;
                                }
                                ;
                                auto tmp24 = tmp19 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                                auto tmp25 = tmp13 ? tmp18 : tmp24;
                                auto tmp26 = tmp4 ? tmp9 : tmp25;
                                out_ptr0[static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (453888L*x1))] = tmp26;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_backward_sum_35 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_36 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_mul_native_layer_norm_backward_sum_37 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_38 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (2364L*x1) + (465708L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (38809L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (12L*x2) + (2364L*x1) + (465708L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (197L*x1) + (38809L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp6.store(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp5;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_put_new_zeros_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8784L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_clone_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(8L))
                            {
                                float tmp16[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), static_cast<long>(197L), tmp16, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                                    auto tmp1 = static_cast<int>(0);
                                    auto tmp2 = tmp0 >= tmp1;
                                    auto tmp3 = static_cast<int>(8);
                                    auto tmp4 = tmp0 < tmp3;
                                    auto tmp5 = [&]
                                    {
                                        auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                        auto tmp7 = static_cast<float>(0.3535533905932738);
                                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                        auto tmp9 = tmp6 * tmp8;
                                        return tmp9;
                                    }
                                    ;
                                    auto tmp10 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                                    auto tmp11 = tmp0 >= tmp3;
                                    auto tmp12 = static_cast<int>(16);
                                    auto tmp13 = tmp0 < tmp12;
                                    auto tmp14 = tmp11 & tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp17 = at::vec::Vectorized<float>::loadu(tmp16 + static_cast<long>(8L*x3_inner));
                                        auto tmp18 = static_cast<float>(0.3535533905932738);
                                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                                        auto tmp20 = tmp17 * tmp19;
                                        return tmp20;
                                    }
                                    ;
                                    auto tmp21 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp14));
                                    auto tmp22 = tmp0 >= tmp12;
                                    auto tmp23 = static_cast<int>(24);
                                    auto tmp24 = tmp0 < tmp23;
                                    auto tmp25 = [&]
                                    {
                                        auto tmp26 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp22));
                                        return tmp26;
                                    }
                                    ;
                                    auto tmp27 = decltype(tmp25())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp25(), to_float_mask(tmp22));
                                    auto tmp28 = to_float_mask(tmp14);
                                    auto tmp29 = decltype(tmp21)::blendv(tmp27, tmp21, tmp28);
                                    auto tmp30 = to_float_mask(tmp4);
                                    auto tmp31 = decltype(tmp10)::blendv(tmp29, tmp10, tmp30);
                                    tmp31.store(out_ptr0 + static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (2304L*x3_inner) + (453888L*x1)));
                                }
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = c10::convert<long>(x1 + (8L*x0));
                                auto tmp1 = static_cast<long>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<long>(8);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = [&]
                                {
                                    auto tmp6 = in_ptr0[static_cast<long>(x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp7 = static_cast<float>(0.3535533905932738);
                                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                    return tmp8;
                                }
                                ;
                                auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                                auto tmp10 = tmp0 >= tmp3;
                                auto tmp11 = static_cast<long>(16);
                                auto tmp12 = tmp0 < tmp11;
                                auto tmp13 = tmp10 & tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr1[static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp16 = static_cast<float>(0.3535533905932738);
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                auto tmp19 = tmp0 >= tmp11;
                                auto tmp20 = static_cast<long>(24);
                                auto tmp21 = tmp0 < tmp20;
                                auto tmp22 = [&]
                                {
                                    auto tmp23 = in_ptr2[static_cast<long>((-2420736L) + x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    return tmp23;
                                }
                                ;
                                auto tmp24 = tmp19 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                                auto tmp25 = tmp13 ? tmp18 : tmp24;
                                auto tmp26 = tmp4 ? tmp9 : tmp25;
                                out_ptr0[static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (453888L*x1))] = tmp26;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_backward_sum_42 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_43 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_mul_native_layer_norm_backward_sum_44 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (2364L*x1) + (465708L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (38809L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (12L*x2) + (2364L*x1) + (465708L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (197L*x1) + (38809L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp6.store(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp5;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_put_new_zeros_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8784L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_clone_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(8L))
                            {
                                float tmp16[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), static_cast<long>(197L), tmp16, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                                    auto tmp1 = static_cast<int>(0);
                                    auto tmp2 = tmp0 >= tmp1;
                                    auto tmp3 = static_cast<int>(8);
                                    auto tmp4 = tmp0 < tmp3;
                                    auto tmp5 = [&]
                                    {
                                        auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                        auto tmp7 = static_cast<float>(0.3535533905932738);
                                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                        auto tmp9 = tmp6 * tmp8;
                                        return tmp9;
                                    }
                                    ;
                                    auto tmp10 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                                    auto tmp11 = tmp0 >= tmp3;
                                    auto tmp12 = static_cast<int>(16);
                                    auto tmp13 = tmp0 < tmp12;
                                    auto tmp14 = tmp11 & tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp17 = at::vec::Vectorized<float>::loadu(tmp16 + static_cast<long>(8L*x3_inner));
                                        auto tmp18 = static_cast<float>(0.3535533905932738);
                                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                                        auto tmp20 = tmp17 * tmp19;
                                        return tmp20;
                                    }
                                    ;
                                    auto tmp21 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp14));
                                    auto tmp22 = tmp0 >= tmp12;
                                    auto tmp23 = static_cast<int>(24);
                                    auto tmp24 = tmp0 < tmp23;
                                    auto tmp25 = [&]
                                    {
                                        auto tmp26 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp22));
                                        return tmp26;
                                    }
                                    ;
                                    auto tmp27 = decltype(tmp25())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp25(), to_float_mask(tmp22));
                                    auto tmp28 = to_float_mask(tmp14);
                                    auto tmp29 = decltype(tmp21)::blendv(tmp27, tmp21, tmp28);
                                    auto tmp30 = to_float_mask(tmp4);
                                    auto tmp31 = decltype(tmp10)::blendv(tmp29, tmp10, tmp30);
                                    tmp31.store(out_ptr0 + static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (2304L*x3_inner) + (453888L*x1)));
                                }
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = c10::convert<long>(x1 + (8L*x0));
                                auto tmp1 = static_cast<long>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<long>(8);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = [&]
                                {
                                    auto tmp6 = in_ptr0[static_cast<long>(x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp7 = static_cast<float>(0.3535533905932738);
                                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                    return tmp8;
                                }
                                ;
                                auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                                auto tmp10 = tmp0 >= tmp3;
                                auto tmp11 = static_cast<long>(16);
                                auto tmp12 = tmp0 < tmp11;
                                auto tmp13 = tmp10 & tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr1[static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp16 = static_cast<float>(0.3535533905932738);
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                auto tmp19 = tmp0 >= tmp11;
                                auto tmp20 = static_cast<long>(24);
                                auto tmp21 = tmp0 < tmp20;
                                auto tmp22 = [&]
                                {
                                    auto tmp23 = in_ptr2[static_cast<long>((-2420736L) + x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    return tmp23;
                                }
                                ;
                                auto tmp24 = tmp19 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                                auto tmp25 = tmp13 ? tmp18 : tmp24;
                                auto tmp26 = tmp4 ? tmp9 : tmp25;
                                out_ptr0[static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (453888L*x1))] = tmp26;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_backward_sum_49 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_50 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_mul_native_layer_norm_backward_sum_51 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_52 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (2364L*x1) + (465708L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (38809L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (12L*x2) + (2364L*x1) + (465708L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (197L*x1) + (38809L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp6.store(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp5;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_put_new_zeros_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8784L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_clone_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(8L))
                            {
                                float tmp16[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), static_cast<long>(197L), tmp16, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                                    auto tmp1 = static_cast<int>(0);
                                    auto tmp2 = tmp0 >= tmp1;
                                    auto tmp3 = static_cast<int>(8);
                                    auto tmp4 = tmp0 < tmp3;
                                    auto tmp5 = [&]
                                    {
                                        auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                        auto tmp7 = static_cast<float>(0.3535533905932738);
                                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                        auto tmp9 = tmp6 * tmp8;
                                        return tmp9;
                                    }
                                    ;
                                    auto tmp10 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                                    auto tmp11 = tmp0 >= tmp3;
                                    auto tmp12 = static_cast<int>(16);
                                    auto tmp13 = tmp0 < tmp12;
                                    auto tmp14 = tmp11 & tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp17 = at::vec::Vectorized<float>::loadu(tmp16 + static_cast<long>(8L*x3_inner));
                                        auto tmp18 = static_cast<float>(0.3535533905932738);
                                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                                        auto tmp20 = tmp17 * tmp19;
                                        return tmp20;
                                    }
                                    ;
                                    auto tmp21 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp14));
                                    auto tmp22 = tmp0 >= tmp12;
                                    auto tmp23 = static_cast<int>(24);
                                    auto tmp24 = tmp0 < tmp23;
                                    auto tmp25 = [&]
                                    {
                                        auto tmp26 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp22));
                                        return tmp26;
                                    }
                                    ;
                                    auto tmp27 = decltype(tmp25())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp25(), to_float_mask(tmp22));
                                    auto tmp28 = to_float_mask(tmp14);
                                    auto tmp29 = decltype(tmp21)::blendv(tmp27, tmp21, tmp28);
                                    auto tmp30 = to_float_mask(tmp4);
                                    auto tmp31 = decltype(tmp10)::blendv(tmp29, tmp10, tmp30);
                                    tmp31.store(out_ptr0 + static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (2304L*x3_inner) + (453888L*x1)));
                                }
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = c10::convert<long>(x1 + (8L*x0));
                                auto tmp1 = static_cast<long>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<long>(8);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = [&]
                                {
                                    auto tmp6 = in_ptr0[static_cast<long>(x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp7 = static_cast<float>(0.3535533905932738);
                                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                    return tmp8;
                                }
                                ;
                                auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                                auto tmp10 = tmp0 >= tmp3;
                                auto tmp11 = static_cast<long>(16);
                                auto tmp12 = tmp0 < tmp11;
                                auto tmp13 = tmp10 & tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr1[static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp16 = static_cast<float>(0.3535533905932738);
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                auto tmp19 = tmp0 >= tmp11;
                                auto tmp20 = static_cast<long>(24);
                                auto tmp21 = tmp0 < tmp20;
                                auto tmp22 = [&]
                                {
                                    auto tmp23 = in_ptr2[static_cast<long>((-2420736L) + x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    return tmp23;
                                }
                                ;
                                auto tmp24 = tmp19 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                                auto tmp25 = tmp13 ? tmp18 : tmp24;
                                auto tmp26 = tmp4 ? tmp9 : tmp25;
                                out_ptr0[static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (453888L*x1))] = tmp26;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_backward_sum_56 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_57 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_mul_native_layer_norm_backward_sum_58 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_59 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (2364L*x1) + (465708L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (38809L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (12L*x2) + (2364L*x1) + (465708L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (197L*x1) + (38809L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp6.store(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp5;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_put_new_zeros_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8784L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_clone_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(8L))
                            {
                                float tmp16[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), static_cast<long>(197L), tmp16, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                                    auto tmp1 = static_cast<int>(0);
                                    auto tmp2 = tmp0 >= tmp1;
                                    auto tmp3 = static_cast<int>(8);
                                    auto tmp4 = tmp0 < tmp3;
                                    auto tmp5 = [&]
                                    {
                                        auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                        auto tmp7 = static_cast<float>(0.3535533905932738);
                                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                        auto tmp9 = tmp6 * tmp8;
                                        return tmp9;
                                    }
                                    ;
                                    auto tmp10 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                                    auto tmp11 = tmp0 >= tmp3;
                                    auto tmp12 = static_cast<int>(16);
                                    auto tmp13 = tmp0 < tmp12;
                                    auto tmp14 = tmp11 & tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp17 = at::vec::Vectorized<float>::loadu(tmp16 + static_cast<long>(8L*x3_inner));
                                        auto tmp18 = static_cast<float>(0.3535533905932738);
                                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                                        auto tmp20 = tmp17 * tmp19;
                                        return tmp20;
                                    }
                                    ;
                                    auto tmp21 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp14));
                                    auto tmp22 = tmp0 >= tmp12;
                                    auto tmp23 = static_cast<int>(24);
                                    auto tmp24 = tmp0 < tmp23;
                                    auto tmp25 = [&]
                                    {
                                        auto tmp26 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp22));
                                        return tmp26;
                                    }
                                    ;
                                    auto tmp27 = decltype(tmp25())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp25(), to_float_mask(tmp22));
                                    auto tmp28 = to_float_mask(tmp14);
                                    auto tmp29 = decltype(tmp21)::blendv(tmp27, tmp21, tmp28);
                                    auto tmp30 = to_float_mask(tmp4);
                                    auto tmp31 = decltype(tmp10)::blendv(tmp29, tmp10, tmp30);
                                    tmp31.store(out_ptr0 + static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (2304L*x3_inner) + (453888L*x1)));
                                }
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = c10::convert<long>(x1 + (8L*x0));
                                auto tmp1 = static_cast<long>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<long>(8);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = [&]
                                {
                                    auto tmp6 = in_ptr0[static_cast<long>(x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp7 = static_cast<float>(0.3535533905932738);
                                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                    return tmp8;
                                }
                                ;
                                auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                                auto tmp10 = tmp0 >= tmp3;
                                auto tmp11 = static_cast<long>(16);
                                auto tmp12 = tmp0 < tmp11;
                                auto tmp13 = tmp10 & tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr1[static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp16 = static_cast<float>(0.3535533905932738);
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                auto tmp19 = tmp0 >= tmp11;
                                auto tmp20 = static_cast<long>(24);
                                auto tmp21 = tmp0 < tmp20;
                                auto tmp22 = [&]
                                {
                                    auto tmp23 = in_ptr2[static_cast<long>((-2420736L) + x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    return tmp23;
                                }
                                ;
                                auto tmp24 = tmp19 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                                auto tmp25 = tmp13 ? tmp18 : tmp24;
                                auto tmp26 = tmp4 ? tmp9 : tmp25;
                                out_ptr0[static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (453888L*x1))] = tmp26;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_backward_sum_63 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_64 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_mul_native_layer_norm_backward_sum_65 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_66 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (2364L*x1) + (465708L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (38809L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (12L*x2) + (2364L*x1) + (465708L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (197L*x1) + (38809L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp6.store(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp5;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_put_new_zeros_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8784L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_clone_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(8L))
                            {
                                float tmp16[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), static_cast<long>(197L), tmp16, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                                    auto tmp1 = static_cast<int>(0);
                                    auto tmp2 = tmp0 >= tmp1;
                                    auto tmp3 = static_cast<int>(8);
                                    auto tmp4 = tmp0 < tmp3;
                                    auto tmp5 = [&]
                                    {
                                        auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                        auto tmp7 = static_cast<float>(0.3535533905932738);
                                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                        auto tmp9 = tmp6 * tmp8;
                                        return tmp9;
                                    }
                                    ;
                                    auto tmp10 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                                    auto tmp11 = tmp0 >= tmp3;
                                    auto tmp12 = static_cast<int>(16);
                                    auto tmp13 = tmp0 < tmp12;
                                    auto tmp14 = tmp11 & tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp17 = at::vec::Vectorized<float>::loadu(tmp16 + static_cast<long>(8L*x3_inner));
                                        auto tmp18 = static_cast<float>(0.3535533905932738);
                                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                                        auto tmp20 = tmp17 * tmp19;
                                        return tmp20;
                                    }
                                    ;
                                    auto tmp21 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp14));
                                    auto tmp22 = tmp0 >= tmp12;
                                    auto tmp23 = static_cast<int>(24);
                                    auto tmp24 = tmp0 < tmp23;
                                    auto tmp25 = [&]
                                    {
                                        auto tmp26 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp22));
                                        return tmp26;
                                    }
                                    ;
                                    auto tmp27 = decltype(tmp25())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp25(), to_float_mask(tmp22));
                                    auto tmp28 = to_float_mask(tmp14);
                                    auto tmp29 = decltype(tmp21)::blendv(tmp27, tmp21, tmp28);
                                    auto tmp30 = to_float_mask(tmp4);
                                    auto tmp31 = decltype(tmp10)::blendv(tmp29, tmp10, tmp30);
                                    tmp31.store(out_ptr0 + static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (2304L*x3_inner) + (453888L*x1)));
                                }
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = c10::convert<long>(x1 + (8L*x0));
                                auto tmp1 = static_cast<long>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<long>(8);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = [&]
                                {
                                    auto tmp6 = in_ptr0[static_cast<long>(x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp7 = static_cast<float>(0.3535533905932738);
                                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                    return tmp8;
                                }
                                ;
                                auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                                auto tmp10 = tmp0 >= tmp3;
                                auto tmp11 = static_cast<long>(16);
                                auto tmp12 = tmp0 < tmp11;
                                auto tmp13 = tmp10 & tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr1[static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp16 = static_cast<float>(0.3535533905932738);
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                auto tmp19 = tmp0 >= tmp11;
                                auto tmp20 = static_cast<long>(24);
                                auto tmp21 = tmp0 < tmp20;
                                auto tmp22 = [&]
                                {
                                    auto tmp23 = in_ptr2[static_cast<long>((-2420736L) + x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    return tmp23;
                                }
                                ;
                                auto tmp24 = tmp19 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                                auto tmp25 = tmp13 ? tmp18 : tmp24;
                                auto tmp26 = tmp4 ? tmp9 : tmp25;
                                out_ptr0[static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (453888L*x1))] = tmp26;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_backward_sum_70 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_71 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_mul_native_layer_norm_backward_sum_72 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_sum_73 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (2364L*x1) + (465708L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (38809L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (12L*x2) + (2364L*x1) + (465708L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (197L*x1) + (38809L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp6.store(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp5;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_index_put_new_zeros_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8784L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_clone_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(8L))
                            {
                                float tmp16[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), static_cast<long>(197L), tmp16, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                                    auto tmp1 = static_cast<int>(0);
                                    auto tmp2 = tmp0 >= tmp1;
                                    auto tmp3 = static_cast<int>(8);
                                    auto tmp4 = tmp0 < tmp3;
                                    auto tmp5 = [&]
                                    {
                                        auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                        auto tmp7 = static_cast<float>(0.3535533905932738);
                                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                        auto tmp9 = tmp6 * tmp8;
                                        return tmp9;
                                    }
                                    ;
                                    auto tmp10 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                                    auto tmp11 = tmp0 >= tmp3;
                                    auto tmp12 = static_cast<int>(16);
                                    auto tmp13 = tmp0 < tmp12;
                                    auto tmp14 = tmp11 & tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp17 = at::vec::Vectorized<float>::loadu(tmp16 + static_cast<long>(8L*x3_inner));
                                        auto tmp18 = static_cast<float>(0.3535533905932738);
                                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                                        auto tmp20 = tmp17 * tmp19;
                                        return tmp20;
                                    }
                                    ;
                                    auto tmp21 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp14));
                                    auto tmp22 = tmp0 >= tmp12;
                                    auto tmp23 = static_cast<int>(24);
                                    auto tmp24 = tmp0 < tmp23;
                                    auto tmp25 = [&]
                                    {
                                        auto tmp26 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp22));
                                        return tmp26;
                                    }
                                    ;
                                    auto tmp27 = decltype(tmp25())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp25(), to_float_mask(tmp22));
                                    auto tmp28 = to_float_mask(tmp14);
                                    auto tmp29 = decltype(tmp21)::blendv(tmp27, tmp21, tmp28);
                                    auto tmp30 = to_float_mask(tmp4);
                                    auto tmp31 = decltype(tmp10)::blendv(tmp29, tmp10, tmp30);
                                    tmp31.store(out_ptr0 + static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (2304L*x3_inner) + (453888L*x1)));
                                }
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = c10::convert<long>(x1 + (8L*x0));
                                auto tmp1 = static_cast<long>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<long>(8);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = [&]
                                {
                                    auto tmp6 = in_ptr0[static_cast<long>(x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp7 = static_cast<float>(0.3535533905932738);
                                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                    return tmp8;
                                }
                                ;
                                auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                                auto tmp10 = tmp0 >= tmp3;
                                auto tmp11 = static_cast<long>(16);
                                auto tmp12 = tmp0 < tmp11;
                                auto tmp13 = tmp10 & tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr1[static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp16 = static_cast<float>(0.3535533905932738);
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                auto tmp19 = tmp0 >= tmp11;
                                auto tmp20 = static_cast<long>(24);
                                auto tmp21 = tmp0 < tmp20;
                                auto tmp22 = [&]
                                {
                                    auto tmp23 = in_ptr2[static_cast<long>((-2420736L) + x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    return tmp23;
                                }
                                ;
                                auto tmp24 = tmp19 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                                auto tmp25 = tmp13 ? tmp18 : tmp24;
                                auto tmp26 = tmp4 ? tmp9 : tmp25;
                                out_ptr0[static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (453888L*x1))] = tmp26;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_native_layer_norm_backward_sum_77 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_mul_native_layer_norm_backward_sum_79 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    auto tmp20 = tmp18 * tmp19;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_sum_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x0 + (12L*x2) + (12L*x2_inner) + (2364L*x1) + (465708L*x3))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = tmp0 * tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp1 * tmp4;
                                auto tmp6 = tmp2 - tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (38809L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x2 + (197L*x1) + (38809L*x0) + (465708L*x3))];
                                auto tmp1 = in_ptr1[static_cast<long>(x0 + (12L*x2) + (2364L*x1) + (465708L*x3))];
                                auto tmp3 = out_ptr0[static_cast<long>(x1 + (197L*x0) + (2364L*x3))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                                auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                                tmp_acc0 = tmp_acc0 + tmp5;
                            }
                            out_ptr1[static_cast<long>(x2 + (197L*x1) + (38809L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2364L*x2) + (465708L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            tmp6.store(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (12L*x3) + (2364L*x2) + (465708L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (465708L*x0))] = tmp5;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_new_zeros_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8784L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_clone_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(3)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(8L))
                            {
                                float tmp16[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), static_cast<long>(197L), tmp16, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                                    auto tmp1 = static_cast<int>(0);
                                    auto tmp2 = tmp0 >= tmp1;
                                    auto tmp3 = static_cast<int>(8);
                                    auto tmp4 = tmp0 < tmp3;
                                    auto tmp5 = [&]
                                    {
                                        auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                        auto tmp7 = static_cast<float>(0.3535533905932738);
                                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                        auto tmp9 = tmp6 * tmp8;
                                        return tmp9;
                                    }
                                    ;
                                    auto tmp10 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                                    auto tmp11 = tmp0 >= tmp3;
                                    auto tmp12 = static_cast<int>(16);
                                    auto tmp13 = tmp0 < tmp12;
                                    auto tmp14 = tmp11 & tmp13;
                                    auto tmp15 = [&]
                                    {
                                        auto tmp17 = at::vec::Vectorized<float>::loadu(tmp16 + static_cast<long>(8L*x3_inner));
                                        auto tmp18 = static_cast<float>(0.3535533905932738);
                                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                                        auto tmp20 = tmp17 * tmp19;
                                        return tmp20;
                                    }
                                    ;
                                    auto tmp21 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp14));
                                    auto tmp22 = tmp0 >= tmp12;
                                    auto tmp23 = static_cast<int>(24);
                                    auto tmp24 = tmp0 < tmp23;
                                    auto tmp25 = [&]
                                    {
                                        auto tmp26 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x4 + (64L*x3) + (64L*x3_inner) + (12608L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp22));
                                        return tmp26;
                                    }
                                    ;
                                    auto tmp27 = decltype(tmp25())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp25(), to_float_mask(tmp22));
                                    auto tmp28 = to_float_mask(tmp14);
                                    auto tmp29 = decltype(tmp21)::blendv(tmp27, tmp21, tmp28);
                                    auto tmp30 = to_float_mask(tmp4);
                                    auto tmp31 = decltype(tmp10)::blendv(tmp29, tmp10, tmp30);
                                    tmp31.store(out_ptr0 + static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (2304L*x3_inner) + (453888L*x1)));
                                }
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(64L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = c10::convert<long>(x1 + (8L*x0));
                                auto tmp1 = static_cast<long>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<long>(8);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = [&]
                                {
                                    auto tmp6 = in_ptr0[static_cast<long>(x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp7 = static_cast<float>(0.3535533905932738);
                                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                    return tmp8;
                                }
                                ;
                                auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                                auto tmp10 = tmp0 >= tmp3;
                                auto tmp11 = static_cast<long>(16);
                                auto tmp12 = tmp0 < tmp11;
                                auto tmp13 = tmp10 & tmp12;
                                auto tmp14 = [&]
                                {
                                    auto tmp15 = in_ptr1[static_cast<long>((-1210368L) + x3 + (197L*x4) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    auto tmp16 = static_cast<float>(0.3535533905932738);
                                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                                    return tmp17;
                                }
                                ;
                                auto tmp18 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                                auto tmp19 = tmp0 >= tmp11;
                                auto tmp20 = static_cast<long>(24);
                                auto tmp21 = tmp0 < tmp20;
                                auto tmp22 = [&]
                                {
                                    auto tmp23 = in_ptr2[static_cast<long>((-2420736L) + x4 + (64L*x3) + (12608L*x2) + (151296L*x1) + (1210368L*x0))];
                                    return tmp23;
                                }
                                ;
                                auto tmp24 = tmp19 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                                auto tmp25 = tmp13 ? tmp18 : tmp24;
                                auto tmp26 = tmp4 ? tmp9 : tmp25;
                                out_ptr0[static_cast<long>(x4 + (64L*x2) + (768L*x0) + (2304L*x3) + (453888L*x1))] = tmp26;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_84 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
                        auto tmp4 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = in_ptr5[static_cast<long>(x0)];
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = tmp2 * tmp9;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = in_ptr4[static_cast<long>(x1)];
                        auto tmp5 = in_ptr5[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = tmp0 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = in_ptr4[static_cast<long>(x0)];
                    auto tmp18 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp2);
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp16 = at::vec::Vectorized<float>(tmp1);
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp17 * tmp19;
                    auto tmp21 = tmp11 - tmp20;
                    auto tmp22 = at::vec::Vectorized<float>(tmp3);
                    auto tmp23 = tmp22 * tmp21;
                    auto tmp24 = tmp0 + tmp23;
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (151296L*x1)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x0));
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
    primals_2, primals_3, primals_9, primals_10, primals_12, primals_13, primals_19, primals_20, primals_22, primals_23, primals_29, primals_30, primals_32, primals_33, primals_39, primals_40, primals_42, primals_43, primals_49, primals_50, primals_52, primals_53, primals_59, primals_60, primals_62, primals_63, primals_69, primals_70, primals_72, primals_73, primals_79, primals_80, primals_82, primals_83, primals_89, primals_90, primals_92, primals_93, primals_99, primals_100, primals_102, primals_103, primals_109, primals_110, primals_112, primals_113, primals_119, primals_120, primals_122, primals_124, primals_224, cat, getitem_1, rsqrt, view_1, view_4, view_13, addmm_1, mul_5, view_15, addmm_2, view_17, addmm_3, mul_11, view_19, view_22, view_31, addmm_5, mul_16, view_33, addmm_6, view_35, addmm_7, mul_22, view_37, view_40, view_49, addmm_9, mul_27, view_51, addmm_10, view_53, addmm_11, mul_33, view_55, view_58, view_67, addmm_13, mul_38, view_69, addmm_14, view_71, addmm_15, mul_44, view_73, view_76, view_85, addmm_17, mul_49, view_87, addmm_18, view_89, addmm_19, mul_55, view_91, view_94, view_103, addmm_21, mul_60, view_105, addmm_22, view_107, addmm_23, mul_66, view_109, view_112, view_121, addmm_25, mul_71, view_123, addmm_26, view_125, addmm_27, mul_77, view_127, view_130, view_139, addmm_29, mul_82, view_141, addmm_30, view_143, addmm_31, mul_88, view_145, view_148, view_157, addmm_33, mul_93, view_159, addmm_34, view_161, addmm_35, mul_99, view_163, view_166, view_175, addmm_37, mul_104, view_177, addmm_38, view_179, addmm_39, mul_110, view_181, view_184, view_193, addmm_41, mul_115, view_195, addmm_42, view_197, addmm_43, mul_121, view_199, view_202, view_211, addmm_45, mul_126, view_213, addmm_46, view_215, addmm_47, mul_132, clone_97, permute_98, div_12, permute_102, permute_106, div_14, permute_110, permute_115, permute_116, alias_12, permute_117, permute_118, permute_122, div_15, permute_126, permute_130, div_16, permute_134, permute_139, permute_140, alias_13, permute_141, permute_142, permute_146, div_17, permute_150, permute_154, div_18, permute_158, permute_163, permute_164, alias_14, permute_165, permute_166, permute_170, div_19, permute_174, permute_178, div_20, permute_182, permute_187, permute_188, alias_15, permute_189, permute_190, permute_194, div_21, permute_198, permute_202, div_22, permute_206, permute_211, permute_212, alias_16, permute_213, permute_214, permute_218, div_23, permute_222, permute_226, div_24, permute_230, permute_235, permute_236, alias_17, permute_237, permute_238, permute_242, div_25, permute_246, permute_250, div_26, permute_254, permute_259, permute_260, alias_18, permute_261, permute_262, permute_266, div_27, permute_270, permute_274, div_28, permute_278, permute_283, permute_284, alias_19, permute_285, permute_286, permute_290, div_29, permute_294, permute_298, div_30, permute_302, permute_307, permute_308, alias_20, permute_309, permute_310, permute_314, div_31, permute_318, permute_322, div_32, permute_326, permute_331, permute_332, alias_21, permute_333, permute_334, permute_338, div_33, permute_342, permute_346, div_34, permute_350, permute_355, permute_356, alias_22, permute_357, permute_358, permute_362, div_35, permute_366, permute_370, div_36, permute_374, permute_379, permute_380, alias_23, permute_381, permute_382, permute_386, tangents_1 = args
    args.clear()
    assert_size_stride(primals_2, (768, ), (1, ))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_50, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (768, ), (1, ))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_92, (768, ), (1, ))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_109, (768, ), (1, ))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (768, ), (1, ))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_122, (768, ), (1, ))
    assert_size_stride(primals_124, (768, 3, 16, 16), (768, 1, 48, 3))
    assert_size_stride(primals_224, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(cat, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(getitem_1, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_1, (1576, 768), (768, 1))
    assert_size_stride(view_4, (38809, ), (1, ))
    assert_size_stride(view_13, (1576, 768), (768, 1))
    assert_size_stride(addmm_1, (1576, 768), (768, 1))
    assert_size_stride(mul_5, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_15, (1576, 768), (768, 1))
    assert_size_stride(addmm_2, (1576, 3072), (3072, 1))
    assert_size_stride(view_17, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_3, (1576, 768), (768, 1))
    assert_size_stride(mul_11, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_19, (1576, 768), (768, 1))
    assert_size_stride(view_22, (38809, ), (1, ))
    assert_size_stride(view_31, (1576, 768), (768, 1))
    assert_size_stride(addmm_5, (1576, 768), (768, 1))
    assert_size_stride(mul_16, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_33, (1576, 768), (768, 1))
    assert_size_stride(addmm_6, (1576, 3072), (3072, 1))
    assert_size_stride(view_35, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_7, (1576, 768), (768, 1))
    assert_size_stride(mul_22, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_37, (1576, 768), (768, 1))
    assert_size_stride(view_40, (38809, ), (1, ))
    assert_size_stride(view_49, (1576, 768), (768, 1))
    assert_size_stride(addmm_9, (1576, 768), (768, 1))
    assert_size_stride(mul_27, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_51, (1576, 768), (768, 1))
    assert_size_stride(addmm_10, (1576, 3072), (3072, 1))
    assert_size_stride(view_53, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_11, (1576, 768), (768, 1))
    assert_size_stride(mul_33, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_55, (1576, 768), (768, 1))
    assert_size_stride(view_58, (38809, ), (1, ))
    assert_size_stride(view_67, (1576, 768), (768, 1))
    assert_size_stride(addmm_13, (1576, 768), (768, 1))
    assert_size_stride(mul_38, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_69, (1576, 768), (768, 1))
    assert_size_stride(addmm_14, (1576, 3072), (3072, 1))
    assert_size_stride(view_71, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_15, (1576, 768), (768, 1))
    assert_size_stride(mul_44, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_73, (1576, 768), (768, 1))
    assert_size_stride(view_76, (38809, ), (1, ))
    assert_size_stride(view_85, (1576, 768), (768, 1))
    assert_size_stride(addmm_17, (1576, 768), (768, 1))
    assert_size_stride(mul_49, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_87, (1576, 768), (768, 1))
    assert_size_stride(addmm_18, (1576, 3072), (3072, 1))
    assert_size_stride(view_89, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_19, (1576, 768), (768, 1))
    assert_size_stride(mul_55, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_91, (1576, 768), (768, 1))
    assert_size_stride(view_94, (38809, ), (1, ))
    assert_size_stride(view_103, (1576, 768), (768, 1))
    assert_size_stride(addmm_21, (1576, 768), (768, 1))
    assert_size_stride(mul_60, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_105, (1576, 768), (768, 1))
    assert_size_stride(addmm_22, (1576, 3072), (3072, 1))
    assert_size_stride(view_107, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_23, (1576, 768), (768, 1))
    assert_size_stride(mul_66, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_109, (1576, 768), (768, 1))
    assert_size_stride(view_112, (38809, ), (1, ))
    assert_size_stride(view_121, (1576, 768), (768, 1))
    assert_size_stride(addmm_25, (1576, 768), (768, 1))
    assert_size_stride(mul_71, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_123, (1576, 768), (768, 1))
    assert_size_stride(addmm_26, (1576, 3072), (3072, 1))
    assert_size_stride(view_125, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_27, (1576, 768), (768, 1))
    assert_size_stride(mul_77, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_127, (1576, 768), (768, 1))
    assert_size_stride(view_130, (38809, ), (1, ))
    assert_size_stride(view_139, (1576, 768), (768, 1))
    assert_size_stride(addmm_29, (1576, 768), (768, 1))
    assert_size_stride(mul_82, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_141, (1576, 768), (768, 1))
    assert_size_stride(addmm_30, (1576, 3072), (3072, 1))
    assert_size_stride(view_143, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_31, (1576, 768), (768, 1))
    assert_size_stride(mul_88, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_145, (1576, 768), (768, 1))
    assert_size_stride(view_148, (38809, ), (1, ))
    assert_size_stride(view_157, (1576, 768), (768, 1))
    assert_size_stride(addmm_33, (1576, 768), (768, 1))
    assert_size_stride(mul_93, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_159, (1576, 768), (768, 1))
    assert_size_stride(addmm_34, (1576, 3072), (3072, 1))
    assert_size_stride(view_161, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_35, (1576, 768), (768, 1))
    assert_size_stride(mul_99, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_163, (1576, 768), (768, 1))
    assert_size_stride(view_166, (38809, ), (1, ))
    assert_size_stride(view_175, (1576, 768), (768, 1))
    assert_size_stride(addmm_37, (1576, 768), (768, 1))
    assert_size_stride(mul_104, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_177, (1576, 768), (768, 1))
    assert_size_stride(addmm_38, (1576, 3072), (3072, 1))
    assert_size_stride(view_179, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_39, (1576, 768), (768, 1))
    assert_size_stride(mul_110, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_181, (1576, 768), (768, 1))
    assert_size_stride(view_184, (38809, ), (1, ))
    assert_size_stride(view_193, (1576, 768), (768, 1))
    assert_size_stride(addmm_41, (1576, 768), (768, 1))
    assert_size_stride(mul_115, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_195, (1576, 768), (768, 1))
    assert_size_stride(addmm_42, (1576, 3072), (3072, 1))
    assert_size_stride(view_197, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_43, (1576, 768), (768, 1))
    assert_size_stride(mul_121, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_199, (1576, 768), (768, 1))
    assert_size_stride(view_202, (38809, ), (1, ))
    assert_size_stride(view_211, (1576, 768), (768, 1))
    assert_size_stride(addmm_45, (1576, 768), (768, 1))
    assert_size_stride(mul_126, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_213, (1576, 768), (768, 1))
    assert_size_stride(addmm_46, (1576, 3072), (3072, 1))
    assert_size_stride(view_215, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_47, (1576, 768), (768, 1))
    assert_size_stride(mul_132, (8, 768), (768, 1))
    assert_size_stride(clone_97, (8, 768), (768, 1))
    assert_size_stride(permute_98, (1000, 768), (768, 1))
    assert_size_stride(div_12, (8, 1), (1, 1))
    assert_size_stride(permute_102, (768, 3072), (3072, 1))
    assert_size_stride(permute_106, (3072, 768), (768, 1))
    assert_size_stride(div_14, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_110, (768, 768), (768, 1))
    assert_size_stride(permute_115, (96, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_116, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_12, (8, 12, 197, 197), (465708, 1, 2364, 12))
    assert_size_stride(permute_117, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_118, (96, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_122, (2304, 768), (768, 1))
    assert_size_stride(div_15, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_126, (768, 3072), (3072, 1))
    assert_size_stride(permute_130, (3072, 768), (768, 1))
    assert_size_stride(div_16, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_134, (768, 768), (768, 1))
    assert_size_stride(permute_139, (96, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_140, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_13, (8, 12, 197, 197), (465708, 1, 2364, 12))
    assert_size_stride(permute_141, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_142, (96, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_146, (2304, 768), (768, 1))
    assert_size_stride(div_17, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_150, (768, 3072), (3072, 1))
    assert_size_stride(permute_154, (3072, 768), (768, 1))
    assert_size_stride(div_18, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_158, (768, 768), (768, 1))
    assert_size_stride(permute_163, (96, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_164, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_14, (8, 12, 197, 197), (465708, 1, 2364, 12))
    assert_size_stride(permute_165, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_166, (96, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_170, (2304, 768), (768, 1))
    assert_size_stride(div_19, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_174, (768, 3072), (3072, 1))
    assert_size_stride(permute_178, (3072, 768), (768, 1))
    assert_size_stride(div_20, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_182, (768, 768), (768, 1))
    assert_size_stride(permute_187, (96, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_188, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_15, (8, 12, 197, 197), (465708, 1, 2364, 12))
    assert_size_stride(permute_189, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_190, (96, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_194, (2304, 768), (768, 1))
    assert_size_stride(div_21, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_198, (768, 3072), (3072, 1))
    assert_size_stride(permute_202, (3072, 768), (768, 1))
    assert_size_stride(div_22, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_206, (768, 768), (768, 1))
    assert_size_stride(permute_211, (96, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_212, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_16, (8, 12, 197, 197), (465708, 1, 2364, 12))
    assert_size_stride(permute_213, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_214, (96, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_218, (2304, 768), (768, 1))
    assert_size_stride(div_23, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_222, (768, 3072), (3072, 1))
    assert_size_stride(permute_226, (3072, 768), (768, 1))
    assert_size_stride(div_24, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_230, (768, 768), (768, 1))
    assert_size_stride(permute_235, (96, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_236, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_17, (8, 12, 197, 197), (465708, 1, 2364, 12))
    assert_size_stride(permute_237, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_238, (96, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_242, (2304, 768), (768, 1))
    assert_size_stride(div_25, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_246, (768, 3072), (3072, 1))
    assert_size_stride(permute_250, (3072, 768), (768, 1))
    assert_size_stride(div_26, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_254, (768, 768), (768, 1))
    assert_size_stride(permute_259, (96, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_260, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_18, (8, 12, 197, 197), (465708, 1, 2364, 12))
    assert_size_stride(permute_261, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_262, (96, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_266, (2304, 768), (768, 1))
    assert_size_stride(div_27, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_270, (768, 3072), (3072, 1))
    assert_size_stride(permute_274, (3072, 768), (768, 1))
    assert_size_stride(div_28, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_278, (768, 768), (768, 1))
    assert_size_stride(permute_283, (96, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_284, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_19, (8, 12, 197, 197), (465708, 1, 2364, 12))
    assert_size_stride(permute_285, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_286, (96, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_290, (2304, 768), (768, 1))
    assert_size_stride(div_29, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_294, (768, 3072), (3072, 1))
    assert_size_stride(permute_298, (3072, 768), (768, 1))
    assert_size_stride(div_30, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_302, (768, 768), (768, 1))
    assert_size_stride(permute_307, (96, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_308, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_20, (8, 12, 197, 197), (465708, 1, 2364, 12))
    assert_size_stride(permute_309, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_310, (96, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_314, (2304, 768), (768, 1))
    assert_size_stride(div_31, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_318, (768, 3072), (3072, 1))
    assert_size_stride(permute_322, (3072, 768), (768, 1))
    assert_size_stride(div_32, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_326, (768, 768), (768, 1))
    assert_size_stride(permute_331, (96, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_332, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_21, (8, 12, 197, 197), (465708, 1, 2364, 12))
    assert_size_stride(permute_333, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_334, (96, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_338, (2304, 768), (768, 1))
    assert_size_stride(div_33, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_342, (768, 3072), (3072, 1))
    assert_size_stride(permute_346, (3072, 768), (768, 1))
    assert_size_stride(div_34, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_350, (768, 768), (768, 1))
    assert_size_stride(permute_355, (96, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_356, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_22, (8, 12, 197, 197), (465708, 1, 2364, 12))
    assert_size_stride(permute_357, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_358, (96, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_362, (2304, 768), (768, 1))
    assert_size_stride(div_35, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_366, (768, 3072), (3072, 1))
    assert_size_stride(permute_370, (3072, 768), (768, 1))
    assert_size_stride(div_36, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_374, (768, 768), (768, 1))
    assert_size_stride(permute_379, (96, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_380, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_23, (8, 12, 197, 197), (465708, 1, 2364, 12))
    assert_size_stride(permute_381, (96, 64, 197), (12608, 1, 64))
    assert_size_stride(permute_382, (96, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_386, (2304, 768), (768, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_98, out=buf0)
    del permute_98
    buf1 = empty((1000, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_97, out=buf1)
    del clone_97
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((8, 1), (1, 8), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 1), (1, 8), device='cpu', dtype=torch.float32)
    buf5 = empty((768, ), device='cpu', dtype=torch.float32)
    buf6 = empty((768, ), device='cpu', dtype=torch.float32)
    buf7 = buf0; del buf0  # reuse
    buf8 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf9 = empty((8, 197, 768), device='cpu', dtype=torch.float32)
    cpp_fused_div_mul_native_layer_norm_backward_slice_backward_sum_0(c_void_p(buf7.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(mul_132.data_ptr()), c_void_p(div_12.data_ptr()), c_void_p(addmm_47.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()))
    del addmm_47
    del buf3
    del buf4
    del div_12
    del mul_132
    del primals_119
    del primals_122
    del tangents_1
    buf10 = empty((1576, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf9, (1576, 768), (768, 1), 0), permute_102, out=buf10)
    del permute_102
    buf11 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf9, (768, 1576), (1, 768), 0), view_215, out=buf11)
    del view_215
    buf12 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf13 = reinterpret_tensor(buf10, (8, 197, 3072), (605184, 3072, 1), 0); del buf10  # reuse
    cpp_fused_gelu_gelu_backward_sum_1(c_void_p(buf13.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(addmm_46.data_ptr()), c_void_p(buf12.data_ptr()))
    del addmm_46
    buf14 = reinterpret_tensor(buf9, (1576, 768), (768, 1), 0); del buf9  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf13, (1576, 3072), (3072, 1), 0), permute_106, out=buf14)
    del permute_106
    buf15 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf13, (3072, 1576), (1, 3072), 0), view_213, out=buf15)
    del view_213
    buf16 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf17 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf18 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf19 = empty((768, ), device='cpu', dtype=torch.float32)
    buf20 = empty((768, ), device='cpu', dtype=torch.float32)
    buf21 = reinterpret_tensor(buf14, (8, 197, 768), (151296, 768, 1), 0); del buf14  # reuse
    buf22 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf23 = empty((8, 197, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_backward_slice_backward_sum_2(c_void_p(buf21.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(mul_126.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(div_14.data_ptr()), c_void_p(addmm_45.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()))
    del addmm_45
    del buf7
    del div_14
    del mul_126
    del primals_112
    del primals_120
    buf24 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf23, (1576, 768), (768, 1), 0), permute_110, out=buf24)
    del permute_110
    buf25 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf23, (768, 1576), (1, 768), 0), view_211, out=buf25)
    del view_211
    buf26 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf27 = empty((8, 12, 197, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_sum_3(c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()))
    buf28 = reinterpret_tensor(buf24, (96, 197, 64), (12608, 64, 1), 0); del buf24  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_115, reinterpret_tensor(buf27, (96, 197, 64), (12608, 64, 1), 0), out=buf28)
    del permute_115
    buf29 = empty((96, 197, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf27, (96, 197, 64), (12608, 64, 1), 0), permute_116, out=buf29)
    del permute_116
    buf30 = empty_strided((8, 12, 197, 1), (2364, 197, 1, 18912), device='cpu', dtype=torch.float32)
    buf31 = empty((1, 12, 197, 197), device='cpu', dtype=torch.float32)
    buf32 = reinterpret_tensor(buf29, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf29  # reuse
    cpp_fused__softmax_backward_data_sum_4(c_void_p(buf32.data_ptr()), c_void_p(alias_12.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    del alias_12
    buf33 = reinterpret_tensor(buf27, (96, 64, 197), (12608, 197, 1), 0); del buf27  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_117, reinterpret_tensor(buf32, (96, 197, 197), (38809, 197, 1), 0), out=buf33)
    del permute_117
    buf34 = reinterpret_tensor(buf23, (96, 197, 64), (12608, 64, 1), 0); del buf23  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf32, (96, 197, 197), (38809, 197, 1), 0), permute_118, out=buf34)
    del permute_118
    buf35 = empty((732, 12), device='cpu', dtype=torch.float32)
    cpp_fused_index_put_new_zeros_5(c_void_p(buf35.data_ptr()))
    aten.index_put_(buf35, [view_202], reinterpret_tensor(buf31, (38809, 12), (1, 38809), 0), True)
    del view_202
    buf38 = empty((8, 197, 3, 12, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_6(c_void_p(buf34.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf38.data_ptr()))
    buf39 = reinterpret_tensor(buf34, (1576, 768), (768, 1), 0); del buf34  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf38, (1576, 2304), (2304, 1), 0), permute_122, out=buf39)
    del permute_122
    buf40 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf38, (2304, 1576), (1, 2304), 0), view_199, out=buf40)
    del view_199
    buf41 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf42 = buf18; del buf18  # reuse
    buf43 = buf17; del buf17  # reuse
    buf44 = empty((768, ), device='cpu', dtype=torch.float32)
    buf45 = empty((768, ), device='cpu', dtype=torch.float32)
    buf46 = buf21; del buf21  # reuse
    buf48 = reinterpret_tensor(buf33, (8, 197, 768), (151296, 768, 1), 0); del buf33  # reuse
    buf47 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_7(c_void_p(buf46.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(mul_121.data_ptr()), c_void_p(div_15.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(addmm_43.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf47.data_ptr()))
    del addmm_43
    del div_15
    del mul_121
    del primals_109
    del primals_113
    buf49 = reinterpret_tensor(buf13, (1576, 3072), (3072, 1), 0); del buf13  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf48, (1576, 768), (768, 1), 0), permute_126, out=buf49)
    del permute_126
    buf50 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf48, (768, 1576), (1, 768), 0), view_197, out=buf50)
    del view_197
    buf51 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf52 = reinterpret_tensor(buf49, (8, 197, 3072), (605184, 3072, 1), 0); del buf49  # reuse
    cpp_fused_gelu_gelu_backward_sum_8(c_void_p(buf52.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(addmm_42.data_ptr()), c_void_p(buf51.data_ptr()))
    del addmm_42
    buf53 = reinterpret_tensor(buf48, (1576, 768), (768, 1), 0); del buf48  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf52, (1576, 3072), (3072, 1), 0), permute_130, out=buf53)
    del permute_130
    buf54 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf52, (3072, 1576), (1, 3072), 0), view_195, out=buf54)
    del view_195
    buf55 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf56 = buf43; del buf43  # reuse
    buf57 = buf42; del buf42  # reuse
    buf58 = empty((768, ), device='cpu', dtype=torch.float32)
    buf59 = empty((768, ), device='cpu', dtype=torch.float32)
    buf60 = buf46; del buf46  # reuse
    buf62 = reinterpret_tensor(buf39, (8, 197, 768), (151296, 768, 1), 0); del buf39  # reuse
    buf61 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_9(c_void_p(buf60.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(mul_115.data_ptr()), c_void_p(div_16.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(addmm_41.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf61.data_ptr()))
    del addmm_41
    del div_16
    del mul_115
    del primals_102
    del primals_110
    buf63 = buf53; del buf53  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf62, (1576, 768), (768, 1), 0), permute_134, out=buf63)
    del permute_134
    buf64 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf62, (768, 1576), (1, 768), 0), view_193, out=buf64)
    del view_193
    buf65 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf66 = reinterpret_tensor(buf28, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf28  # reuse
    cpp_fused_clone_sum_10(c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()))
    buf67 = reinterpret_tensor(buf63, (96, 197, 64), (12608, 64, 1), 0); del buf63  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_139, reinterpret_tensor(buf66, (96, 197, 64), (12608, 64, 1), 0), out=buf67)
    del permute_139
    buf68 = reinterpret_tensor(buf32, (96, 197, 197), (38809, 197, 1), 0); del buf32  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf66, (96, 197, 64), (12608, 64, 1), 0), permute_140, out=buf68)
    del permute_140
    buf69 = buf30; del buf30  # reuse
    buf70 = buf31; del buf31  # reuse
    buf71 = reinterpret_tensor(buf68, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf68  # reuse
    cpp_fused__softmax_backward_data_sum_11(c_void_p(buf71.data_ptr()), c_void_p(alias_13.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    del alias_13
    buf72 = reinterpret_tensor(buf66, (96, 64, 197), (12608, 197, 1), 0); del buf66  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_141, reinterpret_tensor(buf71, (96, 197, 197), (38809, 197, 1), 0), out=buf72)
    del permute_141
    buf73 = reinterpret_tensor(buf62, (96, 197, 64), (12608, 64, 1), 0); del buf62  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf71, (96, 197, 197), (38809, 197, 1), 0), permute_142, out=buf73)
    del permute_142
    buf74 = empty((732, 12), device='cpu', dtype=torch.float32)
    cpp_fused_index_put_new_zeros_12(c_void_p(buf74.data_ptr()))
    aten.index_put_(buf74, [view_184], reinterpret_tensor(buf70, (38809, 12), (1, 38809), 0), True)
    del view_184
    buf77 = buf38; del buf38  # reuse
    cpp_fused_clone_13(c_void_p(buf73.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf77.data_ptr()))
    buf78 = reinterpret_tensor(buf73, (1576, 768), (768, 1), 0); del buf73  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf77, (1576, 2304), (2304, 1), 0), permute_146, out=buf78)
    del permute_146
    buf79 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf77, (2304, 1576), (1, 2304), 0), view_181, out=buf79)
    del view_181
    buf80 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf81 = buf57; del buf57  # reuse
    buf82 = buf56; del buf56  # reuse
    buf83 = empty((768, ), device='cpu', dtype=torch.float32)
    buf84 = empty((768, ), device='cpu', dtype=torch.float32)
    buf85 = buf60; del buf60  # reuse
    buf87 = reinterpret_tensor(buf72, (8, 197, 768), (151296, 768, 1), 0); del buf72  # reuse
    buf86 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_14(c_void_p(buf85.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(mul_110.data_ptr()), c_void_p(div_17.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(addmm_39.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf86.data_ptr()))
    del addmm_39
    del div_17
    del mul_110
    del primals_103
    del primals_99
    buf88 = reinterpret_tensor(buf52, (1576, 3072), (3072, 1), 0); del buf52  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf87, (1576, 768), (768, 1), 0), permute_150, out=buf88)
    del permute_150
    buf89 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf87, (768, 1576), (1, 768), 0), view_179, out=buf89)
    del view_179
    buf90 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf91 = reinterpret_tensor(buf88, (8, 197, 3072), (605184, 3072, 1), 0); del buf88  # reuse
    cpp_fused_gelu_gelu_backward_sum_15(c_void_p(buf91.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(addmm_38.data_ptr()), c_void_p(buf90.data_ptr()))
    del addmm_38
    buf92 = reinterpret_tensor(buf87, (1576, 768), (768, 1), 0); del buf87  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf91, (1576, 3072), (3072, 1), 0), permute_154, out=buf92)
    del permute_154
    buf93 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf91, (3072, 1576), (1, 3072), 0), view_177, out=buf93)
    del view_177
    buf94 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf95 = buf82; del buf82  # reuse
    buf96 = buf81; del buf81  # reuse
    buf97 = empty((768, ), device='cpu', dtype=torch.float32)
    buf98 = empty((768, ), device='cpu', dtype=torch.float32)
    buf99 = buf85; del buf85  # reuse
    buf101 = reinterpret_tensor(buf78, (8, 197, 768), (151296, 768, 1), 0); del buf78  # reuse
    buf100 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_16(c_void_p(buf99.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(mul_104.data_ptr()), c_void_p(div_18.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(addmm_37.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf100.data_ptr()))
    del addmm_37
    del div_18
    del mul_104
    del primals_100
    del primals_92
    buf102 = buf92; del buf92  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf101, (1576, 768), (768, 1), 0), permute_158, out=buf102)
    del permute_158
    buf103 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf101, (768, 1576), (1, 768), 0), view_175, out=buf103)
    del view_175
    buf104 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf105 = reinterpret_tensor(buf67, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf67  # reuse
    cpp_fused_clone_sum_17(c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()))
    buf106 = reinterpret_tensor(buf102, (96, 197, 64), (12608, 64, 1), 0); del buf102  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_163, reinterpret_tensor(buf105, (96, 197, 64), (12608, 64, 1), 0), out=buf106)
    del permute_163
    buf107 = reinterpret_tensor(buf71, (96, 197, 197), (38809, 197, 1), 0); del buf71  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf105, (96, 197, 64), (12608, 64, 1), 0), permute_164, out=buf107)
    del permute_164
    buf108 = buf69; del buf69  # reuse
    buf109 = buf70; del buf70  # reuse
    buf110 = reinterpret_tensor(buf107, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf107  # reuse
    cpp_fused__softmax_backward_data_sum_18(c_void_p(buf110.data_ptr()), c_void_p(alias_14.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    del alias_14
    buf111 = reinterpret_tensor(buf105, (96, 64, 197), (12608, 197, 1), 0); del buf105  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_165, reinterpret_tensor(buf110, (96, 197, 197), (38809, 197, 1), 0), out=buf111)
    del permute_165
    buf112 = reinterpret_tensor(buf101, (96, 197, 64), (12608, 64, 1), 0); del buf101  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf110, (96, 197, 197), (38809, 197, 1), 0), permute_166, out=buf112)
    del permute_166
    buf113 = empty((732, 12), device='cpu', dtype=torch.float32)
    cpp_fused_index_put_new_zeros_19(c_void_p(buf113.data_ptr()))
    aten.index_put_(buf113, [view_166], reinterpret_tensor(buf109, (38809, 12), (1, 38809), 0), True)
    del view_166
    buf116 = buf77; del buf77  # reuse
    cpp_fused_clone_20(c_void_p(buf112.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf116.data_ptr()))
    buf117 = reinterpret_tensor(buf112, (1576, 768), (768, 1), 0); del buf112  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf116, (1576, 2304), (2304, 1), 0), permute_170, out=buf117)
    del permute_170
    buf118 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf116, (2304, 1576), (1, 2304), 0), view_163, out=buf118)
    del view_163
    buf119 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf120 = buf96; del buf96  # reuse
    buf121 = buf95; del buf95  # reuse
    buf122 = empty((768, ), device='cpu', dtype=torch.float32)
    buf123 = empty((768, ), device='cpu', dtype=torch.float32)
    buf124 = reinterpret_tensor(buf117, (8, 197, 768), (151296, 768, 1), 0); del buf117  # reuse
    buf126 = reinterpret_tensor(buf111, (8, 197, 768), (151296, 768, 1), 0); del buf111  # reuse
    buf125 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_21(c_void_p(buf124.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(mul_99.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(div_19.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(addmm_35.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf125.data_ptr()))
    del addmm_35
    del div_19
    del mul_99
    del primals_89
    del primals_93
    buf127 = reinterpret_tensor(buf91, (1576, 3072), (3072, 1), 0); del buf91  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf126, (1576, 768), (768, 1), 0), permute_174, out=buf127)
    del permute_174
    buf128 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf126, (768, 1576), (1, 768), 0), view_161, out=buf128)
    del view_161
    buf129 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf130 = reinterpret_tensor(buf127, (8, 197, 3072), (605184, 3072, 1), 0); del buf127  # reuse
    cpp_fused_gelu_gelu_backward_sum_22(c_void_p(buf130.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(buf129.data_ptr()))
    del addmm_34
    buf131 = reinterpret_tensor(buf126, (1576, 768), (768, 1), 0); del buf126  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf130, (1576, 3072), (3072, 1), 0), permute_178, out=buf131)
    del permute_178
    buf132 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf130, (3072, 1576), (1, 3072), 0), view_159, out=buf132)
    del view_159
    buf133 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf134 = buf121; del buf121  # reuse
    buf135 = buf120; del buf120  # reuse
    buf136 = empty((768, ), device='cpu', dtype=torch.float32)
    buf137 = empty((768, ), device='cpu', dtype=torch.float32)
    buf138 = buf124; del buf124  # reuse
    buf140 = buf99; del buf99  # reuse
    buf139 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_23(c_void_p(buf138.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(mul_93.data_ptr()), c_void_p(div_20.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(addmm_33.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf139.data_ptr()))
    del addmm_33
    del div_20
    del mul_93
    del primals_82
    del primals_90
    buf141 = buf131; del buf131  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf140, (1576, 768), (768, 1), 0), permute_182, out=buf141)
    del permute_182
    buf142 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf140, (768, 1576), (1, 768), 0), view_157, out=buf142)
    del view_157
    buf143 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf144 = reinterpret_tensor(buf106, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf106  # reuse
    cpp_fused_clone_sum_24(c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()))
    buf145 = reinterpret_tensor(buf141, (96, 197, 64), (12608, 64, 1), 0); del buf141  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_187, reinterpret_tensor(buf144, (96, 197, 64), (12608, 64, 1), 0), out=buf145)
    del permute_187
    buf146 = reinterpret_tensor(buf110, (96, 197, 197), (38809, 197, 1), 0); del buf110  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf144, (96, 197, 64), (12608, 64, 1), 0), permute_188, out=buf146)
    del permute_188
    buf147 = buf108; del buf108  # reuse
    buf148 = buf109; del buf109  # reuse
    buf149 = reinterpret_tensor(buf146, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf146  # reuse
    cpp_fused__softmax_backward_data_sum_25(c_void_p(buf149.data_ptr()), c_void_p(alias_15.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()))
    del alias_15
    buf150 = reinterpret_tensor(buf144, (96, 64, 197), (12608, 197, 1), 0); del buf144  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_189, reinterpret_tensor(buf149, (96, 197, 197), (38809, 197, 1), 0), out=buf150)
    del permute_189
    buf151 = reinterpret_tensor(buf140, (96, 197, 64), (12608, 64, 1), 0); del buf140  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf149, (96, 197, 197), (38809, 197, 1), 0), permute_190, out=buf151)
    del permute_190
    buf152 = empty((732, 12), device='cpu', dtype=torch.float32)
    cpp_fused_index_put_new_zeros_26(c_void_p(buf152.data_ptr()))
    aten.index_put_(buf152, [view_148], reinterpret_tensor(buf148, (38809, 12), (1, 38809), 0), True)
    del view_148
    buf155 = buf116; del buf116  # reuse
    cpp_fused_clone_27(c_void_p(buf151.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf155.data_ptr()))
    buf156 = reinterpret_tensor(buf151, (1576, 768), (768, 1), 0); del buf151  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf155, (1576, 2304), (2304, 1), 0), permute_194, out=buf156)
    del permute_194
    buf157 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf155, (2304, 1576), (1, 2304), 0), view_145, out=buf157)
    del view_145
    buf158 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf159 = buf135; del buf135  # reuse
    buf160 = buf134; del buf134  # reuse
    buf161 = empty((768, ), device='cpu', dtype=torch.float32)
    buf162 = empty((768, ), device='cpu', dtype=torch.float32)
    buf163 = buf138; del buf138  # reuse
    buf165 = reinterpret_tensor(buf150, (8, 197, 768), (151296, 768, 1), 0); del buf150  # reuse
    buf164 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_28(c_void_p(buf163.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(mul_88.data_ptr()), c_void_p(div_21.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(addmm_31.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf164.data_ptr()))
    del addmm_31
    del div_21
    del mul_88
    del primals_79
    del primals_83
    buf166 = reinterpret_tensor(buf130, (1576, 3072), (3072, 1), 0); del buf130  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf165, (1576, 768), (768, 1), 0), permute_198, out=buf166)
    del permute_198
    buf167 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf165, (768, 1576), (1, 768), 0), view_143, out=buf167)
    del view_143
    buf168 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf169 = reinterpret_tensor(buf166, (8, 197, 3072), (605184, 3072, 1), 0); del buf166  # reuse
    cpp_fused_gelu_gelu_backward_sum_29(c_void_p(buf169.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(addmm_30.data_ptr()), c_void_p(buf168.data_ptr()))
    del addmm_30
    buf170 = reinterpret_tensor(buf165, (1576, 768), (768, 1), 0); del buf165  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf169, (1576, 3072), (3072, 1), 0), permute_202, out=buf170)
    del permute_202
    buf171 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf169, (3072, 1576), (1, 3072), 0), view_141, out=buf171)
    del view_141
    buf172 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf173 = buf160; del buf160  # reuse
    buf174 = buf159; del buf159  # reuse
    buf175 = empty((768, ), device='cpu', dtype=torch.float32)
    buf176 = empty((768, ), device='cpu', dtype=torch.float32)
    buf177 = buf163; del buf163  # reuse
    buf179 = reinterpret_tensor(buf156, (8, 197, 768), (151296, 768, 1), 0); del buf156  # reuse
    buf178 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_30(c_void_p(buf177.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(mul_82.data_ptr()), c_void_p(div_22.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(addmm_29.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf178.data_ptr()))
    del addmm_29
    del div_22
    del mul_82
    del primals_72
    del primals_80
    buf180 = buf170; del buf170  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf179, (1576, 768), (768, 1), 0), permute_206, out=buf180)
    del permute_206
    buf181 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf179, (768, 1576), (1, 768), 0), view_139, out=buf181)
    del view_139
    buf182 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf183 = reinterpret_tensor(buf145, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf145  # reuse
    cpp_fused_clone_sum_31(c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()))
    buf184 = reinterpret_tensor(buf180, (96, 197, 64), (12608, 64, 1), 0); del buf180  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_211, reinterpret_tensor(buf183, (96, 197, 64), (12608, 64, 1), 0), out=buf184)
    del permute_211
    buf185 = reinterpret_tensor(buf149, (96, 197, 197), (38809, 197, 1), 0); del buf149  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf183, (96, 197, 64), (12608, 64, 1), 0), permute_212, out=buf185)
    del permute_212
    buf186 = buf147; del buf147  # reuse
    buf187 = buf148; del buf148  # reuse
    buf188 = reinterpret_tensor(buf185, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf185  # reuse
    cpp_fused__softmax_backward_data_sum_32(c_void_p(buf188.data_ptr()), c_void_p(alias_16.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()))
    del alias_16
    buf189 = reinterpret_tensor(buf183, (96, 64, 197), (12608, 197, 1), 0); del buf183  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_213, reinterpret_tensor(buf188, (96, 197, 197), (38809, 197, 1), 0), out=buf189)
    del permute_213
    buf190 = reinterpret_tensor(buf179, (96, 197, 64), (12608, 64, 1), 0); del buf179  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf188, (96, 197, 197), (38809, 197, 1), 0), permute_214, out=buf190)
    del permute_214
    buf191 = empty((732, 12), device='cpu', dtype=torch.float32)
    cpp_fused_index_put_new_zeros_33(c_void_p(buf191.data_ptr()))
    aten.index_put_(buf191, [view_130], reinterpret_tensor(buf187, (38809, 12), (1, 38809), 0), True)
    del view_130
    buf194 = buf155; del buf155  # reuse
    cpp_fused_clone_34(c_void_p(buf190.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf194.data_ptr()))
    buf195 = reinterpret_tensor(buf190, (1576, 768), (768, 1), 0); del buf190  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf194, (1576, 2304), (2304, 1), 0), permute_218, out=buf195)
    del permute_218
    buf196 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf194, (2304, 1576), (1, 2304), 0), view_127, out=buf196)
    del view_127
    buf197 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf198 = buf174; del buf174  # reuse
    buf199 = buf173; del buf173  # reuse
    buf200 = empty((768, ), device='cpu', dtype=torch.float32)
    buf201 = empty((768, ), device='cpu', dtype=torch.float32)
    buf202 = buf177; del buf177  # reuse
    buf204 = reinterpret_tensor(buf189, (8, 197, 768), (151296, 768, 1), 0); del buf189  # reuse
    buf203 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_35(c_void_p(buf202.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(mul_77.data_ptr()), c_void_p(div_23.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(addmm_27.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf203.data_ptr()))
    del addmm_27
    del div_23
    del mul_77
    del primals_69
    del primals_73
    buf205 = reinterpret_tensor(buf169, (1576, 3072), (3072, 1), 0); del buf169  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf204, (1576, 768), (768, 1), 0), permute_222, out=buf205)
    del permute_222
    buf206 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf204, (768, 1576), (1, 768), 0), view_125, out=buf206)
    del view_125
    buf207 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf208 = reinterpret_tensor(buf205, (8, 197, 3072), (605184, 3072, 1), 0); del buf205  # reuse
    cpp_fused_gelu_gelu_backward_sum_36(c_void_p(buf208.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(addmm_26.data_ptr()), c_void_p(buf207.data_ptr()))
    del addmm_26
    buf209 = reinterpret_tensor(buf204, (1576, 768), (768, 1), 0); del buf204  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf208, (1576, 3072), (3072, 1), 0), permute_226, out=buf209)
    del permute_226
    buf210 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf208, (3072, 1576), (1, 3072), 0), view_123, out=buf210)
    del view_123
    buf211 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf212 = buf199; del buf199  # reuse
    buf213 = buf198; del buf198  # reuse
    buf214 = empty((768, ), device='cpu', dtype=torch.float32)
    buf215 = empty((768, ), device='cpu', dtype=torch.float32)
    buf216 = buf202; del buf202  # reuse
    buf218 = reinterpret_tensor(buf195, (8, 197, 768), (151296, 768, 1), 0); del buf195  # reuse
    buf217 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_37(c_void_p(buf216.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(mul_71.data_ptr()), c_void_p(div_24.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(addmm_25.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf217.data_ptr()))
    del addmm_25
    del div_24
    del mul_71
    del primals_62
    del primals_70
    buf219 = buf209; del buf209  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf218, (1576, 768), (768, 1), 0), permute_230, out=buf219)
    del permute_230
    buf220 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf218, (768, 1576), (1, 768), 0), view_121, out=buf220)
    del view_121
    buf221 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf222 = reinterpret_tensor(buf184, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf184  # reuse
    cpp_fused_clone_sum_38(c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()))
    buf223 = reinterpret_tensor(buf219, (96, 197, 64), (12608, 64, 1), 0); del buf219  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_235, reinterpret_tensor(buf222, (96, 197, 64), (12608, 64, 1), 0), out=buf223)
    del permute_235
    buf224 = reinterpret_tensor(buf188, (96, 197, 197), (38809, 197, 1), 0); del buf188  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf222, (96, 197, 64), (12608, 64, 1), 0), permute_236, out=buf224)
    del permute_236
    buf225 = buf186; del buf186  # reuse
    buf226 = buf187; del buf187  # reuse
    buf227 = reinterpret_tensor(buf224, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf224  # reuse
    cpp_fused__softmax_backward_data_sum_39(c_void_p(buf227.data_ptr()), c_void_p(alias_17.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()))
    del alias_17
    buf228 = reinterpret_tensor(buf222, (96, 64, 197), (12608, 197, 1), 0); del buf222  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_237, reinterpret_tensor(buf227, (96, 197, 197), (38809, 197, 1), 0), out=buf228)
    del permute_237
    buf229 = reinterpret_tensor(buf218, (96, 197, 64), (12608, 64, 1), 0); del buf218  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf227, (96, 197, 197), (38809, 197, 1), 0), permute_238, out=buf229)
    del permute_238
    buf230 = empty((732, 12), device='cpu', dtype=torch.float32)
    cpp_fused_index_put_new_zeros_40(c_void_p(buf230.data_ptr()))
    aten.index_put_(buf230, [view_112], reinterpret_tensor(buf226, (38809, 12), (1, 38809), 0), True)
    del view_112
    buf233 = buf194; del buf194  # reuse
    cpp_fused_clone_41(c_void_p(buf229.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf233.data_ptr()))
    buf234 = reinterpret_tensor(buf229, (1576, 768), (768, 1), 0); del buf229  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf233, (1576, 2304), (2304, 1), 0), permute_242, out=buf234)
    del permute_242
    buf235 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf233, (2304, 1576), (1, 2304), 0), view_109, out=buf235)
    del view_109
    buf236 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf237 = buf213; del buf213  # reuse
    buf238 = buf212; del buf212  # reuse
    buf239 = empty((768, ), device='cpu', dtype=torch.float32)
    buf240 = empty((768, ), device='cpu', dtype=torch.float32)
    buf241 = buf216; del buf216  # reuse
    buf243 = reinterpret_tensor(buf228, (8, 197, 768), (151296, 768, 1), 0); del buf228  # reuse
    buf242 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_42(c_void_p(buf241.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(mul_66.data_ptr()), c_void_p(div_25.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(addmm_23.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf242.data_ptr()))
    del addmm_23
    del div_25
    del mul_66
    del primals_59
    del primals_63
    buf244 = reinterpret_tensor(buf208, (1576, 3072), (3072, 1), 0); del buf208  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf243, (1576, 768), (768, 1), 0), permute_246, out=buf244)
    del permute_246
    buf245 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf243, (768, 1576), (1, 768), 0), view_107, out=buf245)
    del view_107
    buf246 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf247 = reinterpret_tensor(buf244, (8, 197, 3072), (605184, 3072, 1), 0); del buf244  # reuse
    cpp_fused_gelu_gelu_backward_sum_43(c_void_p(buf247.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(buf246.data_ptr()))
    del addmm_22
    buf248 = reinterpret_tensor(buf243, (1576, 768), (768, 1), 0); del buf243  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf247, (1576, 3072), (3072, 1), 0), permute_250, out=buf248)
    del permute_250
    buf249 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf247, (3072, 1576), (1, 3072), 0), view_105, out=buf249)
    del view_105
    buf250 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf251 = buf238; del buf238  # reuse
    buf252 = buf237; del buf237  # reuse
    buf253 = empty((768, ), device='cpu', dtype=torch.float32)
    buf254 = empty((768, ), device='cpu', dtype=torch.float32)
    buf255 = buf241; del buf241  # reuse
    buf257 = reinterpret_tensor(buf234, (8, 197, 768), (151296, 768, 1), 0); del buf234  # reuse
    buf256 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_44(c_void_p(buf255.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(mul_60.data_ptr()), c_void_p(div_26.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(addmm_21.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf256.data_ptr()))
    del addmm_21
    del div_26
    del mul_60
    del primals_52
    del primals_60
    buf258 = buf248; del buf248  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf257, (1576, 768), (768, 1), 0), permute_254, out=buf258)
    del permute_254
    buf259 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf257, (768, 1576), (1, 768), 0), view_103, out=buf259)
    del view_103
    buf260 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf261 = reinterpret_tensor(buf223, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf223  # reuse
    cpp_fused_clone_sum_45(c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()))
    buf262 = reinterpret_tensor(buf258, (96, 197, 64), (12608, 64, 1), 0); del buf258  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_259, reinterpret_tensor(buf261, (96, 197, 64), (12608, 64, 1), 0), out=buf262)
    del permute_259
    buf263 = reinterpret_tensor(buf227, (96, 197, 197), (38809, 197, 1), 0); del buf227  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf261, (96, 197, 64), (12608, 64, 1), 0), permute_260, out=buf263)
    del permute_260
    buf264 = buf225; del buf225  # reuse
    buf265 = buf226; del buf226  # reuse
    buf266 = reinterpret_tensor(buf263, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf263  # reuse
    cpp_fused__softmax_backward_data_sum_46(c_void_p(buf266.data_ptr()), c_void_p(alias_18.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()))
    del alias_18
    buf267 = reinterpret_tensor(buf261, (96, 64, 197), (12608, 197, 1), 0); del buf261  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_261, reinterpret_tensor(buf266, (96, 197, 197), (38809, 197, 1), 0), out=buf267)
    del permute_261
    buf268 = reinterpret_tensor(buf257, (96, 197, 64), (12608, 64, 1), 0); del buf257  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf266, (96, 197, 197), (38809, 197, 1), 0), permute_262, out=buf268)
    del permute_262
    buf269 = empty((732, 12), device='cpu', dtype=torch.float32)
    cpp_fused_index_put_new_zeros_47(c_void_p(buf269.data_ptr()))
    aten.index_put_(buf269, [view_94], reinterpret_tensor(buf265, (38809, 12), (1, 38809), 0), True)
    del view_94
    buf272 = buf233; del buf233  # reuse
    cpp_fused_clone_48(c_void_p(buf268.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf272.data_ptr()))
    buf273 = reinterpret_tensor(buf268, (1576, 768), (768, 1), 0); del buf268  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf272, (1576, 2304), (2304, 1), 0), permute_266, out=buf273)
    del permute_266
    buf274 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf272, (2304, 1576), (1, 2304), 0), view_91, out=buf274)
    del view_91
    buf275 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf276 = buf252; del buf252  # reuse
    buf277 = buf251; del buf251  # reuse
    buf278 = empty((768, ), device='cpu', dtype=torch.float32)
    buf279 = empty((768, ), device='cpu', dtype=torch.float32)
    buf280 = buf255; del buf255  # reuse
    buf282 = reinterpret_tensor(buf267, (8, 197, 768), (151296, 768, 1), 0); del buf267  # reuse
    buf281 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_49(c_void_p(buf280.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(mul_55.data_ptr()), c_void_p(div_27.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(addmm_19.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf281.data_ptr()))
    del addmm_19
    del div_27
    del mul_55
    del primals_49
    del primals_53
    buf283 = reinterpret_tensor(buf247, (1576, 3072), (3072, 1), 0); del buf247  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf282, (1576, 768), (768, 1), 0), permute_270, out=buf283)
    del permute_270
    buf284 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf282, (768, 1576), (1, 768), 0), view_89, out=buf284)
    del view_89
    buf285 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf286 = reinterpret_tensor(buf283, (8, 197, 3072), (605184, 3072, 1), 0); del buf283  # reuse
    cpp_fused_gelu_gelu_backward_sum_50(c_void_p(buf286.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(addmm_18.data_ptr()), c_void_p(buf285.data_ptr()))
    del addmm_18
    buf287 = reinterpret_tensor(buf282, (1576, 768), (768, 1), 0); del buf282  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf286, (1576, 3072), (3072, 1), 0), permute_274, out=buf287)
    del permute_274
    buf288 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf286, (3072, 1576), (1, 3072), 0), view_87, out=buf288)
    del view_87
    buf289 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf290 = buf277; del buf277  # reuse
    buf291 = buf276; del buf276  # reuse
    buf292 = empty((768, ), device='cpu', dtype=torch.float32)
    buf293 = empty((768, ), device='cpu', dtype=torch.float32)
    buf294 = buf280; del buf280  # reuse
    buf296 = reinterpret_tensor(buf273, (8, 197, 768), (151296, 768, 1), 0); del buf273  # reuse
    buf295 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_51(c_void_p(buf294.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(mul_49.data_ptr()), c_void_p(div_28.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(addmm_17.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf295.data_ptr()))
    del addmm_17
    del div_28
    del mul_49
    del primals_42
    del primals_50
    buf297 = buf287; del buf287  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf296, (1576, 768), (768, 1), 0), permute_278, out=buf297)
    del permute_278
    buf298 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf296, (768, 1576), (1, 768), 0), view_85, out=buf298)
    del view_85
    buf299 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf300 = reinterpret_tensor(buf262, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf262  # reuse
    cpp_fused_clone_sum_52(c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()))
    buf301 = reinterpret_tensor(buf297, (96, 197, 64), (12608, 64, 1), 0); del buf297  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_283, reinterpret_tensor(buf300, (96, 197, 64), (12608, 64, 1), 0), out=buf301)
    del permute_283
    buf302 = reinterpret_tensor(buf266, (96, 197, 197), (38809, 197, 1), 0); del buf266  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf300, (96, 197, 64), (12608, 64, 1), 0), permute_284, out=buf302)
    del permute_284
    buf303 = buf264; del buf264  # reuse
    buf304 = buf265; del buf265  # reuse
    buf305 = reinterpret_tensor(buf302, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf302  # reuse
    cpp_fused__softmax_backward_data_sum_53(c_void_p(buf305.data_ptr()), c_void_p(alias_19.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()))
    del alias_19
    buf306 = reinterpret_tensor(buf300, (96, 64, 197), (12608, 197, 1), 0); del buf300  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_285, reinterpret_tensor(buf305, (96, 197, 197), (38809, 197, 1), 0), out=buf306)
    del permute_285
    buf307 = reinterpret_tensor(buf296, (96, 197, 64), (12608, 64, 1), 0); del buf296  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf305, (96, 197, 197), (38809, 197, 1), 0), permute_286, out=buf307)
    del permute_286
    buf308 = empty((732, 12), device='cpu', dtype=torch.float32)
    cpp_fused_index_put_new_zeros_54(c_void_p(buf308.data_ptr()))
    aten.index_put_(buf308, [view_76], reinterpret_tensor(buf304, (38809, 12), (1, 38809), 0), True)
    del view_76
    buf311 = buf272; del buf272  # reuse
    cpp_fused_clone_55(c_void_p(buf307.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf311.data_ptr()))
    buf312 = reinterpret_tensor(buf307, (1576, 768), (768, 1), 0); del buf307  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf311, (1576, 2304), (2304, 1), 0), permute_290, out=buf312)
    del permute_290
    buf313 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf311, (2304, 1576), (1, 2304), 0), view_73, out=buf313)
    del view_73
    buf314 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf315 = buf291; del buf291  # reuse
    buf316 = buf290; del buf290  # reuse
    buf317 = empty((768, ), device='cpu', dtype=torch.float32)
    buf318 = empty((768, ), device='cpu', dtype=torch.float32)
    buf319 = buf294; del buf294  # reuse
    buf321 = reinterpret_tensor(buf306, (8, 197, 768), (151296, 768, 1), 0); del buf306  # reuse
    buf320 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_56(c_void_p(buf319.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(mul_44.data_ptr()), c_void_p(div_29.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(addmm_15.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf320.data_ptr()))
    del addmm_15
    del div_29
    del mul_44
    del primals_39
    del primals_43
    buf322 = reinterpret_tensor(buf286, (1576, 3072), (3072, 1), 0); del buf286  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf321, (1576, 768), (768, 1), 0), permute_294, out=buf322)
    del permute_294
    buf323 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf321, (768, 1576), (1, 768), 0), view_71, out=buf323)
    del view_71
    buf324 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf325 = reinterpret_tensor(buf322, (8, 197, 3072), (605184, 3072, 1), 0); del buf322  # reuse
    cpp_fused_gelu_gelu_backward_sum_57(c_void_p(buf325.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(addmm_14.data_ptr()), c_void_p(buf324.data_ptr()))
    del addmm_14
    buf326 = reinterpret_tensor(buf321, (1576, 768), (768, 1), 0); del buf321  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf325, (1576, 3072), (3072, 1), 0), permute_298, out=buf326)
    del permute_298
    buf327 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf325, (3072, 1576), (1, 3072), 0), view_69, out=buf327)
    del view_69
    buf328 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf329 = buf316; del buf316  # reuse
    buf330 = buf315; del buf315  # reuse
    buf331 = empty((768, ), device='cpu', dtype=torch.float32)
    buf332 = empty((768, ), device='cpu', dtype=torch.float32)
    buf333 = buf319; del buf319  # reuse
    buf335 = reinterpret_tensor(buf312, (8, 197, 768), (151296, 768, 1), 0); del buf312  # reuse
    buf334 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_58(c_void_p(buf333.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(mul_38.data_ptr()), c_void_p(div_30.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(addmm_13.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf334.data_ptr()))
    del addmm_13
    del div_30
    del mul_38
    del primals_32
    del primals_40
    buf336 = buf326; del buf326  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf335, (1576, 768), (768, 1), 0), permute_302, out=buf336)
    del permute_302
    buf337 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf335, (768, 1576), (1, 768), 0), view_67, out=buf337)
    del view_67
    buf338 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf339 = reinterpret_tensor(buf301, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf301  # reuse
    cpp_fused_clone_sum_59(c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()))
    buf340 = reinterpret_tensor(buf336, (96, 197, 64), (12608, 64, 1), 0); del buf336  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_307, reinterpret_tensor(buf339, (96, 197, 64), (12608, 64, 1), 0), out=buf340)
    del permute_307
    buf341 = reinterpret_tensor(buf305, (96, 197, 197), (38809, 197, 1), 0); del buf305  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf339, (96, 197, 64), (12608, 64, 1), 0), permute_308, out=buf341)
    del permute_308
    buf342 = buf303; del buf303  # reuse
    buf343 = buf304; del buf304  # reuse
    buf344 = reinterpret_tensor(buf341, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf341  # reuse
    cpp_fused__softmax_backward_data_sum_60(c_void_p(buf344.data_ptr()), c_void_p(alias_20.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()))
    del alias_20
    buf345 = reinterpret_tensor(buf339, (96, 64, 197), (12608, 197, 1), 0); del buf339  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_309, reinterpret_tensor(buf344, (96, 197, 197), (38809, 197, 1), 0), out=buf345)
    del permute_309
    buf346 = reinterpret_tensor(buf335, (96, 197, 64), (12608, 64, 1), 0); del buf335  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf344, (96, 197, 197), (38809, 197, 1), 0), permute_310, out=buf346)
    del permute_310
    buf347 = empty((732, 12), device='cpu', dtype=torch.float32)
    cpp_fused_index_put_new_zeros_61(c_void_p(buf347.data_ptr()))
    aten.index_put_(buf347, [view_58], reinterpret_tensor(buf343, (38809, 12), (1, 38809), 0), True)
    del view_58
    buf350 = buf311; del buf311  # reuse
    cpp_fused_clone_62(c_void_p(buf346.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf350.data_ptr()))
    buf351 = reinterpret_tensor(buf346, (1576, 768), (768, 1), 0); del buf346  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf350, (1576, 2304), (2304, 1), 0), permute_314, out=buf351)
    del permute_314
    buf352 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf350, (2304, 1576), (1, 2304), 0), view_55, out=buf352)
    del view_55
    buf353 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf354 = buf330; del buf330  # reuse
    buf355 = buf329; del buf329  # reuse
    buf356 = empty((768, ), device='cpu', dtype=torch.float32)
    buf357 = empty((768, ), device='cpu', dtype=torch.float32)
    buf358 = buf333; del buf333  # reuse
    buf360 = reinterpret_tensor(buf345, (8, 197, 768), (151296, 768, 1), 0); del buf345  # reuse
    buf359 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_63(c_void_p(buf358.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(mul_33.data_ptr()), c_void_p(div_31.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(addmm_11.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf359.data_ptr()))
    del addmm_11
    del div_31
    del mul_33
    del primals_29
    del primals_33
    buf361 = reinterpret_tensor(buf325, (1576, 3072), (3072, 1), 0); del buf325  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf360, (1576, 768), (768, 1), 0), permute_318, out=buf361)
    del permute_318
    buf362 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf360, (768, 1576), (1, 768), 0), view_53, out=buf362)
    del view_53
    buf363 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf364 = reinterpret_tensor(buf361, (8, 197, 3072), (605184, 3072, 1), 0); del buf361  # reuse
    cpp_fused_gelu_gelu_backward_sum_64(c_void_p(buf364.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(buf363.data_ptr()))
    del addmm_10
    buf365 = reinterpret_tensor(buf360, (1576, 768), (768, 1), 0); del buf360  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf364, (1576, 3072), (3072, 1), 0), permute_322, out=buf365)
    del permute_322
    buf366 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf364, (3072, 1576), (1, 3072), 0), view_51, out=buf366)
    del view_51
    buf367 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf368 = buf355; del buf355  # reuse
    buf369 = buf354; del buf354  # reuse
    buf370 = empty((768, ), device='cpu', dtype=torch.float32)
    buf371 = empty((768, ), device='cpu', dtype=torch.float32)
    buf372 = buf358; del buf358  # reuse
    buf374 = reinterpret_tensor(buf351, (8, 197, 768), (151296, 768, 1), 0); del buf351  # reuse
    buf373 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_65(c_void_p(buf372.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(mul_27.data_ptr()), c_void_p(div_32.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(addmm_9.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf373.data_ptr()))
    del addmm_9
    del div_32
    del mul_27
    del primals_22
    del primals_30
    buf375 = buf365; del buf365  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf374, (1576, 768), (768, 1), 0), permute_326, out=buf375)
    del permute_326
    buf376 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf374, (768, 1576), (1, 768), 0), view_49, out=buf376)
    del view_49
    buf377 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf378 = reinterpret_tensor(buf340, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf340  # reuse
    cpp_fused_clone_sum_66(c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()))
    buf379 = reinterpret_tensor(buf375, (96, 197, 64), (12608, 64, 1), 0); del buf375  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_331, reinterpret_tensor(buf378, (96, 197, 64), (12608, 64, 1), 0), out=buf379)
    del permute_331
    buf380 = reinterpret_tensor(buf344, (96, 197, 197), (38809, 197, 1), 0); del buf344  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf378, (96, 197, 64), (12608, 64, 1), 0), permute_332, out=buf380)
    del permute_332
    buf381 = buf342; del buf342  # reuse
    buf382 = buf343; del buf343  # reuse
    buf383 = reinterpret_tensor(buf380, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf380  # reuse
    cpp_fused__softmax_backward_data_sum_67(c_void_p(buf383.data_ptr()), c_void_p(alias_21.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()))
    del alias_21
    buf384 = reinterpret_tensor(buf378, (96, 64, 197), (12608, 197, 1), 0); del buf378  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_333, reinterpret_tensor(buf383, (96, 197, 197), (38809, 197, 1), 0), out=buf384)
    del permute_333
    buf385 = reinterpret_tensor(buf374, (96, 197, 64), (12608, 64, 1), 0); del buf374  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf383, (96, 197, 197), (38809, 197, 1), 0), permute_334, out=buf385)
    del permute_334
    buf386 = empty((732, 12), device='cpu', dtype=torch.float32)
    cpp_fused_index_put_new_zeros_68(c_void_p(buf386.data_ptr()))
    aten.index_put_(buf386, [view_40], reinterpret_tensor(buf382, (38809, 12), (1, 38809), 0), True)
    del view_40
    buf389 = buf350; del buf350  # reuse
    cpp_fused_clone_69(c_void_p(buf385.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf389.data_ptr()))
    buf390 = reinterpret_tensor(buf385, (1576, 768), (768, 1), 0); del buf385  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf389, (1576, 2304), (2304, 1), 0), permute_338, out=buf390)
    del permute_338
    buf391 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf389, (2304, 1576), (1, 2304), 0), view_37, out=buf391)
    del view_37
    buf392 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf393 = buf369; del buf369  # reuse
    buf394 = buf368; del buf368  # reuse
    buf395 = empty((768, ), device='cpu', dtype=torch.float32)
    buf396 = empty((768, ), device='cpu', dtype=torch.float32)
    buf397 = buf372; del buf372  # reuse
    buf399 = reinterpret_tensor(buf384, (8, 197, 768), (151296, 768, 1), 0); del buf384  # reuse
    buf398 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_70(c_void_p(buf397.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(mul_22.data_ptr()), c_void_p(div_33.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(addmm_7.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf398.data_ptr()))
    del addmm_7
    del div_33
    del mul_22
    del primals_19
    del primals_23
    buf400 = reinterpret_tensor(buf364, (1576, 3072), (3072, 1), 0); del buf364  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf399, (1576, 768), (768, 1), 0), permute_342, out=buf400)
    del permute_342
    buf401 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf399, (768, 1576), (1, 768), 0), view_35, out=buf401)
    del view_35
    buf402 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf403 = reinterpret_tensor(buf400, (8, 197, 3072), (605184, 3072, 1), 0); del buf400  # reuse
    cpp_fused_gelu_gelu_backward_sum_71(c_void_p(buf403.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(addmm_6.data_ptr()), c_void_p(buf402.data_ptr()))
    del addmm_6
    buf404 = reinterpret_tensor(buf399, (1576, 768), (768, 1), 0); del buf399  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf403, (1576, 3072), (3072, 1), 0), permute_346, out=buf404)
    del permute_346
    buf405 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf403, (3072, 1576), (1, 3072), 0), view_33, out=buf405)
    del view_33
    buf406 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf407 = buf394; del buf394  # reuse
    buf408 = buf393; del buf393  # reuse
    buf409 = empty((768, ), device='cpu', dtype=torch.float32)
    buf410 = empty((768, ), device='cpu', dtype=torch.float32)
    buf411 = buf397; del buf397  # reuse
    buf413 = reinterpret_tensor(buf390, (8, 197, 768), (151296, 768, 1), 0); del buf390  # reuse
    buf412 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_72(c_void_p(buf411.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(mul_16.data_ptr()), c_void_p(div_34.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(addmm_5.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf412.data_ptr()))
    del addmm_5
    del div_34
    del mul_16
    del primals_12
    del primals_20
    buf414 = buf404; del buf404  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf413, (1576, 768), (768, 1), 0), permute_350, out=buf414)
    del permute_350
    buf415 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf413, (768, 1576), (1, 768), 0), view_31, out=buf415)
    del view_31
    buf416 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf417 = reinterpret_tensor(buf379, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf379  # reuse
    cpp_fused_clone_sum_73(c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()))
    buf418 = reinterpret_tensor(buf414, (96, 197, 64), (12608, 64, 1), 0); del buf414  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_355, reinterpret_tensor(buf417, (96, 197, 64), (12608, 64, 1), 0), out=buf418)
    del permute_355
    buf419 = reinterpret_tensor(buf383, (96, 197, 197), (38809, 197, 1), 0); del buf383  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf417, (96, 197, 64), (12608, 64, 1), 0), permute_356, out=buf419)
    del permute_356
    buf420 = buf381; del buf381  # reuse
    buf421 = buf382; del buf382  # reuse
    buf422 = reinterpret_tensor(buf419, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf419  # reuse
    cpp_fused__softmax_backward_data_sum_74(c_void_p(buf422.data_ptr()), c_void_p(alias_22.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()))
    del alias_22
    buf423 = reinterpret_tensor(buf417, (96, 64, 197), (12608, 197, 1), 0); del buf417  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_357, reinterpret_tensor(buf422, (96, 197, 197), (38809, 197, 1), 0), out=buf423)
    del permute_357
    buf424 = reinterpret_tensor(buf413, (96, 197, 64), (12608, 64, 1), 0); del buf413  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf422, (96, 197, 197), (38809, 197, 1), 0), permute_358, out=buf424)
    del permute_358
    buf425 = empty((732, 12), device='cpu', dtype=torch.float32)
    cpp_fused_index_put_new_zeros_75(c_void_p(buf425.data_ptr()))
    aten.index_put_(buf425, [view_22], reinterpret_tensor(buf421, (38809, 12), (1, 38809), 0), True)
    del view_22
    buf428 = buf389; del buf389  # reuse
    cpp_fused_clone_76(c_void_p(buf424.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf428.data_ptr()))
    buf429 = reinterpret_tensor(buf424, (1576, 768), (768, 1), 0); del buf424  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf428, (1576, 2304), (2304, 1), 0), permute_362, out=buf429)
    del permute_362
    buf430 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf428, (2304, 1576), (1, 2304), 0), view_19, out=buf430)
    del view_19
    buf431 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf432 = buf408; del buf408  # reuse
    buf433 = buf407; del buf407  # reuse
    buf434 = empty((768, ), device='cpu', dtype=torch.float32)
    buf435 = empty((768, ), device='cpu', dtype=torch.float32)
    buf436 = buf411; del buf411  # reuse
    buf438 = reinterpret_tensor(buf423, (8, 197, 768), (151296, 768, 1), 0); del buf423  # reuse
    buf437 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_77(c_void_p(buf436.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(mul_11.data_ptr()), c_void_p(div_35.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(addmm_3.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf437.data_ptr()))
    del addmm_3
    del div_35
    del mul_11
    del primals_13
    del primals_9
    buf439 = reinterpret_tensor(buf403, (1576, 3072), (3072, 1), 0); del buf403  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf438, (1576, 768), (768, 1), 0), permute_366, out=buf439)
    del permute_366
    buf440 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf438, (768, 1576), (1, 768), 0), view_17, out=buf440)
    del view_17
    buf441 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf442 = reinterpret_tensor(buf439, (8, 197, 3072), (605184, 3072, 1), 0); del buf439  # reuse
    cpp_fused_gelu_gelu_backward_sum_78(c_void_p(buf442.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(addmm_2.data_ptr()), c_void_p(buf441.data_ptr()))
    del addmm_2
    buf443 = reinterpret_tensor(buf438, (1576, 768), (768, 1), 0); del buf438  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf442, (1576, 3072), (3072, 1), 0), permute_370, out=buf443)
    del permute_370
    buf444 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf442, (3072, 1576), (1, 3072), 0), view_15, out=buf444)
    del view_15
    buf445 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf446 = buf433; del buf433  # reuse
    buf447 = buf432; del buf432  # reuse
    buf448 = empty((768, ), device='cpu', dtype=torch.float32)
    buf449 = empty((768, ), device='cpu', dtype=torch.float32)
    buf450 = buf436; del buf436  # reuse
    buf452 = reinterpret_tensor(buf429, (8, 197, 768), (151296, 768, 1), 0); del buf429  # reuse
    buf451 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_native_layer_norm_backward_sum_79(c_void_p(buf450.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(mul_5.data_ptr()), c_void_p(div_36.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(addmm_1.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf451.data_ptr()))
    del addmm_1
    del buf442
    del div_36
    del mul_5
    del primals_10
    del primals_2
    buf453 = buf443; del buf443  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf452, (1576, 768), (768, 1), 0), permute_374, out=buf453)
    del permute_374
    buf454 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf452, (768, 1576), (1, 768), 0), view_13, out=buf454)
    del view_13
    buf455 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf456 = reinterpret_tensor(buf418, (8, 12, 197, 64), (151296, 12608, 64, 1), 0); del buf418  # reuse
    cpp_fused_clone_sum_80(c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()))
    buf457 = reinterpret_tensor(buf453, (96, 197, 64), (12608, 64, 1), 0); del buf453  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_379, reinterpret_tensor(buf456, (96, 197, 64), (12608, 64, 1), 0), out=buf457)
    del permute_379
    buf458 = reinterpret_tensor(buf422, (96, 197, 197), (38809, 197, 1), 0); del buf422  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf456, (96, 197, 64), (12608, 64, 1), 0), permute_380, out=buf458)
    del permute_380
    buf459 = buf420; del buf420  # reuse
    buf460 = buf421; del buf421  # reuse
    buf461 = reinterpret_tensor(buf458, (8, 12, 197, 197), (465708, 38809, 197, 1), 0); del buf458  # reuse
    cpp_fused__softmax_backward_data_sum_81(c_void_p(buf461.data_ptr()), c_void_p(alias_23.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf460.data_ptr()))
    del alias_23
    del buf459
    buf462 = reinterpret_tensor(buf456, (96, 64, 197), (12608, 197, 1), 0); del buf456  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_381, reinterpret_tensor(buf461, (96, 197, 197), (38809, 197, 1), 0), out=buf462)
    del permute_381
    buf463 = reinterpret_tensor(buf452, (96, 197, 64), (12608, 64, 1), 0); del buf452  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf461, (96, 197, 197), (38809, 197, 1), 0), permute_382, out=buf463)
    del buf461
    del permute_382
    buf464 = empty((732, 12), device='cpu', dtype=torch.float32)
    cpp_fused_new_zeros_82(c_void_p(buf464.data_ptr()))
    aten.index_put_(buf464, [view_4], reinterpret_tensor(buf460, (38809, 12), (1, 38809), 0), True)
    del buf460
    del view_4
    buf467 = buf428; del buf428  # reuse
    cpp_fused_clone_83(c_void_p(buf463.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf467.data_ptr()))
    del buf457
    del buf462
    buf468 = reinterpret_tensor(buf463, (1576, 768), (768, 1), 0); del buf463  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf467, (1576, 2304), (2304, 1), 0), permute_386, out=buf468)
    del permute_386
    buf469 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf467, (2304, 1576), (1, 2304), 0), view_1, out=buf469)
    del view_1
    buf470 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf471 = buf447; del buf447  # reuse
    buf472 = buf446; del buf446  # reuse
    buf473 = empty((768, ), device='cpu', dtype=torch.float32)
    buf474 = empty((768, ), device='cpu', dtype=torch.float32)
    buf475 = buf450; del buf450  # reuse
    buf476 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_84(c_void_p(buf475.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(cat.data_ptr()), c_void_p(getitem_1.data_ptr()), c_void_p(rsqrt.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf476.data_ptr()))
    del buf467
    del buf468
    del buf471
    del buf472
    del cat
    del getitem_1
    del primals_3
    del rsqrt
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf477 = aten.convolution_backward(reinterpret_tensor(buf475, (8, 768, 14, 14), (151296, 1, 10752, 768), 768), primals_224, primals_124, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf475
    del primals_124
    del primals_224
    buf478 = buf477[1]
    buf479 = buf477[2]
    return (buf476, reinterpret_tensor(buf451, (768, ), (1, ), 0), buf473, buf474, reinterpret_tensor(buf470, (768, ), (1, ), 0), reinterpret_tensor(buf470, (768, ), (1, ), 1536), reinterpret_tensor(buf469, (2304, 768), (768, 1), 0), buf464, reinterpret_tensor(buf437, (768, ), (1, ), 0), buf448, buf449, reinterpret_tensor(buf412, (768, ), (1, ), 0), buf434, buf435, reinterpret_tensor(buf431, (768, ), (1, ), 0), reinterpret_tensor(buf431, (768, ), (1, ), 1536), reinterpret_tensor(buf430, (2304, 768), (768, 1), 0), buf425, reinterpret_tensor(buf398, (768, ), (1, ), 0), buf409, buf410, reinterpret_tensor(buf373, (768, ), (1, ), 0), buf395, buf396, reinterpret_tensor(buf392, (768, ), (1, ), 0), reinterpret_tensor(buf392, (768, ), (1, ), 1536), reinterpret_tensor(buf391, (2304, 768), (768, 1), 0), buf386, reinterpret_tensor(buf359, (768, ), (1, ), 0), buf370, buf371, reinterpret_tensor(buf334, (768, ), (1, ), 0), buf356, buf357, reinterpret_tensor(buf353, (768, ), (1, ), 0), reinterpret_tensor(buf353, (768, ), (1, ), 1536), reinterpret_tensor(buf352, (2304, 768), (768, 1), 0), buf347, reinterpret_tensor(buf320, (768, ), (1, ), 0), buf331, buf332, reinterpret_tensor(buf295, (768, ), (1, ), 0), buf317, buf318, reinterpret_tensor(buf314, (768, ), (1, ), 0), reinterpret_tensor(buf314, (768, ), (1, ), 1536), reinterpret_tensor(buf313, (2304, 768), (768, 1), 0), buf308, reinterpret_tensor(buf281, (768, ), (1, ), 0), buf292, buf293, reinterpret_tensor(buf256, (768, ), (1, ), 0), buf278, buf279, reinterpret_tensor(buf275, (768, ), (1, ), 0), reinterpret_tensor(buf275, (768, ), (1, ), 1536), reinterpret_tensor(buf274, (2304, 768), (768, 1), 0), buf269, reinterpret_tensor(buf242, (768, ), (1, ), 0), buf253, buf254, reinterpret_tensor(buf217, (768, ), (1, ), 0), buf239, buf240, reinterpret_tensor(buf236, (768, ), (1, ), 0), reinterpret_tensor(buf236, (768, ), (1, ), 1536), reinterpret_tensor(buf235, (2304, 768), (768, 1), 0), buf230, reinterpret_tensor(buf203, (768, ), (1, ), 0), buf214, buf215, reinterpret_tensor(buf178, (768, ), (1, ), 0), buf200, buf201, reinterpret_tensor(buf197, (768, ), (1, ), 0), reinterpret_tensor(buf197, (768, ), (1, ), 1536), reinterpret_tensor(buf196, (2304, 768), (768, 1), 0), buf191, reinterpret_tensor(buf164, (768, ), (1, ), 0), buf175, buf176, reinterpret_tensor(buf139, (768, ), (1, ), 0), buf161, buf162, reinterpret_tensor(buf158, (768, ), (1, ), 0), reinterpret_tensor(buf158, (768, ), (1, ), 1536), reinterpret_tensor(buf157, (2304, 768), (768, 1), 0), buf152, reinterpret_tensor(buf125, (768, ), (1, ), 0), buf136, buf137, reinterpret_tensor(buf100, (768, ), (1, ), 0), buf122, buf123, reinterpret_tensor(buf119, (768, ), (1, ), 0), reinterpret_tensor(buf119, (768, ), (1, ), 1536), reinterpret_tensor(buf118, (2304, 768), (768, 1), 0), buf113, reinterpret_tensor(buf86, (768, ), (1, ), 0), buf97, buf98, reinterpret_tensor(buf61, (768, ), (1, ), 0), buf83, buf84, reinterpret_tensor(buf80, (768, ), (1, ), 0), reinterpret_tensor(buf80, (768, ), (1, ), 1536), reinterpret_tensor(buf79, (2304, 768), (768, 1), 0), buf74, reinterpret_tensor(buf47, (768, ), (1, ), 0), buf58, buf59, reinterpret_tensor(buf22, (768, ), (1, ), 0), buf44, buf45, reinterpret_tensor(buf41, (768, ), (1, ), 0), reinterpret_tensor(buf41, (768, ), (1, ), 1536), reinterpret_tensor(buf40, (2304, 768), (768, 1), 0), buf35, reinterpret_tensor(buf8, (768, ), (1, ), 0), buf19, buf20, buf5, buf6, buf478, buf479, reinterpret_tensor(buf454, (768, 768), (768, 1), 0), reinterpret_tensor(buf455, (768, ), (1, ), 0), reinterpret_tensor(buf444, (3072, 768), (768, 1), 0), reinterpret_tensor(buf445, (3072, ), (1, ), 0), reinterpret_tensor(buf440, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf441, (768, ), (1, ), 0), reinterpret_tensor(buf415, (768, 768), (768, 1), 0), reinterpret_tensor(buf416, (768, ), (1, ), 0), reinterpret_tensor(buf405, (3072, 768), (768, 1), 0), reinterpret_tensor(buf406, (3072, ), (1, ), 0), reinterpret_tensor(buf401, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf402, (768, ), (1, ), 0), reinterpret_tensor(buf376, (768, 768), (768, 1), 0), reinterpret_tensor(buf377, (768, ), (1, ), 0), reinterpret_tensor(buf366, (3072, 768), (768, 1), 0), reinterpret_tensor(buf367, (3072, ), (1, ), 0), reinterpret_tensor(buf362, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf363, (768, ), (1, ), 0), reinterpret_tensor(buf337, (768, 768), (768, 1), 0), reinterpret_tensor(buf338, (768, ), (1, ), 0), reinterpret_tensor(buf327, (3072, 768), (768, 1), 0), reinterpret_tensor(buf328, (3072, ), (1, ), 0), reinterpret_tensor(buf323, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf324, (768, ), (1, ), 0), reinterpret_tensor(buf298, (768, 768), (768, 1), 0), reinterpret_tensor(buf299, (768, ), (1, ), 0), reinterpret_tensor(buf288, (3072, 768), (768, 1), 0), reinterpret_tensor(buf289, (3072, ), (1, ), 0), reinterpret_tensor(buf284, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf285, (768, ), (1, ), 0), reinterpret_tensor(buf259, (768, 768), (768, 1), 0), reinterpret_tensor(buf260, (768, ), (1, ), 0), reinterpret_tensor(buf249, (3072, 768), (768, 1), 0), reinterpret_tensor(buf250, (3072, ), (1, ), 0), reinterpret_tensor(buf245, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf246, (768, ), (1, ), 0), reinterpret_tensor(buf220, (768, 768), (768, 1), 0), reinterpret_tensor(buf221, (768, ), (1, ), 0), reinterpret_tensor(buf210, (3072, 768), (768, 1), 0), reinterpret_tensor(buf211, (3072, ), (1, ), 0), reinterpret_tensor(buf206, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf207, (768, ), (1, ), 0), reinterpret_tensor(buf181, (768, 768), (768, 1), 0), reinterpret_tensor(buf182, (768, ), (1, ), 0), reinterpret_tensor(buf171, (3072, 768), (768, 1), 0), reinterpret_tensor(buf172, (3072, ), (1, ), 0), reinterpret_tensor(buf167, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf168, (768, ), (1, ), 0), reinterpret_tensor(buf142, (768, 768), (768, 1), 0), reinterpret_tensor(buf143, (768, ), (1, ), 0), reinterpret_tensor(buf132, (3072, 768), (768, 1), 0), reinterpret_tensor(buf133, (3072, ), (1, ), 0), reinterpret_tensor(buf128, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf129, (768, ), (1, ), 0), reinterpret_tensor(buf103, (768, 768), (768, 1), 0), reinterpret_tensor(buf104, (768, ), (1, ), 0), reinterpret_tensor(buf93, (3072, 768), (768, 1), 0), reinterpret_tensor(buf94, (3072, ), (1, ), 0), reinterpret_tensor(buf89, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf90, (768, ), (1, ), 0), reinterpret_tensor(buf64, (768, 768), (768, 1), 0), reinterpret_tensor(buf65, (768, ), (1, ), 0), reinterpret_tensor(buf54, (3072, 768), (768, 1), 0), reinterpret_tensor(buf55, (3072, ), (1, ), 0), reinterpret_tensor(buf50, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf51, (768, ), (1, ), 0), reinterpret_tensor(buf25, (768, 768), (768, 1), 0), reinterpret_tensor(buf26, (768, ), (1, ), 0), reinterpret_tensor(buf15, (3072, 768), (768, 1), 0), reinterpret_tensor(buf16, (3072, ), (1, ), 0), reinterpret_tensor(buf11, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf12, (768, ), (1, ), 0), reinterpret_tensor(buf1, (1000, 768), (768, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((768, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cat = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    getitem_1 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    view_1 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_4 = rand_strided((38809, ), (1, ), device='cpu', dtype=torch.int64)
    view_13 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_1 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_5 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_15 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_2 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_17 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    addmm_3 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_11 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_19 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_22 = rand_strided((38809, ), (1, ), device='cpu', dtype=torch.int64)
    view_31 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_5 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_16 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_33 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_6 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_35 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    addmm_7 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_22 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_37 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_40 = rand_strided((38809, ), (1, ), device='cpu', dtype=torch.int64)
    view_49 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_9 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_27 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_51 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_53 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    addmm_11 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_33 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_55 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_58 = rand_strided((38809, ), (1, ), device='cpu', dtype=torch.int64)
    view_67 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_13 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_38 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_69 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_14 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_71 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    addmm_15 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_44 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_73 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_76 = rand_strided((38809, ), (1, ), device='cpu', dtype=torch.int64)
    view_85 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_17 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_49 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_87 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_18 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_89 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    addmm_19 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_55 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_91 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_94 = rand_strided((38809, ), (1, ), device='cpu', dtype=torch.int64)
    view_103 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_21 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_60 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_105 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_107 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    addmm_23 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_66 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_109 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_112 = rand_strided((38809, ), (1, ), device='cpu', dtype=torch.int64)
    view_121 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_25 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_71 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_123 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_26 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_125 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    addmm_27 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_77 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_127 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_130 = rand_strided((38809, ), (1, ), device='cpu', dtype=torch.int64)
    view_139 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_29 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_82 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_141 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_30 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_143 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    addmm_31 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_88 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_145 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_148 = rand_strided((38809, ), (1, ), device='cpu', dtype=torch.int64)
    view_157 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_33 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_93 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_159 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_161 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    addmm_35 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_99 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_163 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_166 = rand_strided((38809, ), (1, ), device='cpu', dtype=torch.int64)
    view_175 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_37 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_104 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_177 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_38 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_179 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    addmm_39 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_110 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_181 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_184 = rand_strided((38809, ), (1, ), device='cpu', dtype=torch.int64)
    view_193 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_41 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_115 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_195 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_42 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_197 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    addmm_43 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_121 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_199 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_202 = rand_strided((38809, ), (1, ), device='cpu', dtype=torch.int64)
    view_211 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_45 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_126 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_213 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_46 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_215 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    addmm_47 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_132 = rand_strided((8, 768), (768, 1), device='cpu', dtype=torch.float32)
    clone_97 = rand_strided((8, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_98 = rand_strided((1000, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_12 = rand_strided((8, 1), (1, 1), device='cpu', dtype=torch.float32)
    permute_102 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_106 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_14 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_110 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_115 = rand_strided((96, 197, 197), (38809, 1, 197), device='cpu', dtype=torch.float32)
    permute_116 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    alias_12 = rand_strided((8, 12, 197, 197), (465708, 1, 2364, 12), device='cpu', dtype=torch.float32)
    permute_117 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    permute_118 = rand_strided((96, 197, 64), (12608, 1, 197), device='cpu', dtype=torch.float32)
    permute_122 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_15 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_126 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_130 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_16 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_134 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_139 = rand_strided((96, 197, 197), (38809, 1, 197), device='cpu', dtype=torch.float32)
    permute_140 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    alias_13 = rand_strided((8, 12, 197, 197), (465708, 1, 2364, 12), device='cpu', dtype=torch.float32)
    permute_141 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    permute_142 = rand_strided((96, 197, 64), (12608, 1, 197), device='cpu', dtype=torch.float32)
    permute_146 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_17 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_150 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_154 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_18 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_158 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_163 = rand_strided((96, 197, 197), (38809, 1, 197), device='cpu', dtype=torch.float32)
    permute_164 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    alias_14 = rand_strided((8, 12, 197, 197), (465708, 1, 2364, 12), device='cpu', dtype=torch.float32)
    permute_165 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    permute_166 = rand_strided((96, 197, 64), (12608, 1, 197), device='cpu', dtype=torch.float32)
    permute_170 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_19 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_174 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_178 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_20 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_182 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_187 = rand_strided((96, 197, 197), (38809, 1, 197), device='cpu', dtype=torch.float32)
    permute_188 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    alias_15 = rand_strided((8, 12, 197, 197), (465708, 1, 2364, 12), device='cpu', dtype=torch.float32)
    permute_189 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    permute_190 = rand_strided((96, 197, 64), (12608, 1, 197), device='cpu', dtype=torch.float32)
    permute_194 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_21 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_198 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_202 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_22 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_206 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_211 = rand_strided((96, 197, 197), (38809, 1, 197), device='cpu', dtype=torch.float32)
    permute_212 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    alias_16 = rand_strided((8, 12, 197, 197), (465708, 1, 2364, 12), device='cpu', dtype=torch.float32)
    permute_213 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    permute_214 = rand_strided((96, 197, 64), (12608, 1, 197), device='cpu', dtype=torch.float32)
    permute_218 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_23 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_222 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_226 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_230 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_235 = rand_strided((96, 197, 197), (38809, 1, 197), device='cpu', dtype=torch.float32)
    permute_236 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    alias_17 = rand_strided((8, 12, 197, 197), (465708, 1, 2364, 12), device='cpu', dtype=torch.float32)
    permute_237 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    permute_238 = rand_strided((96, 197, 64), (12608, 1, 197), device='cpu', dtype=torch.float32)
    permute_242 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_25 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_246 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_250 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_26 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_254 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_259 = rand_strided((96, 197, 197), (38809, 1, 197), device='cpu', dtype=torch.float32)
    permute_260 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    alias_18 = rand_strided((8, 12, 197, 197), (465708, 1, 2364, 12), device='cpu', dtype=torch.float32)
    permute_261 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    permute_262 = rand_strided((96, 197, 64), (12608, 1, 197), device='cpu', dtype=torch.float32)
    permute_266 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_27 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_270 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_274 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_28 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_278 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_283 = rand_strided((96, 197, 197), (38809, 1, 197), device='cpu', dtype=torch.float32)
    permute_284 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    alias_19 = rand_strided((8, 12, 197, 197), (465708, 1, 2364, 12), device='cpu', dtype=torch.float32)
    permute_285 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    permute_286 = rand_strided((96, 197, 64), (12608, 1, 197), device='cpu', dtype=torch.float32)
    permute_290 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_29 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_294 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_298 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_30 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_302 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_307 = rand_strided((96, 197, 197), (38809, 1, 197), device='cpu', dtype=torch.float32)
    permute_308 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    alias_20 = rand_strided((8, 12, 197, 197), (465708, 1, 2364, 12), device='cpu', dtype=torch.float32)
    permute_309 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    permute_310 = rand_strided((96, 197, 64), (12608, 1, 197), device='cpu', dtype=torch.float32)
    permute_314 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_31 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_318 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_322 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_32 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_326 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_331 = rand_strided((96, 197, 197), (38809, 1, 197), device='cpu', dtype=torch.float32)
    permute_332 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    alias_21 = rand_strided((8, 12, 197, 197), (465708, 1, 2364, 12), device='cpu', dtype=torch.float32)
    permute_333 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    permute_334 = rand_strided((96, 197, 64), (12608, 1, 197), device='cpu', dtype=torch.float32)
    permute_338 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_33 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_342 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_346 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_34 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_350 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_355 = rand_strided((96, 197, 197), (38809, 1, 197), device='cpu', dtype=torch.float32)
    permute_356 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    alias_22 = rand_strided((8, 12, 197, 197), (465708, 1, 2364, 12), device='cpu', dtype=torch.float32)
    permute_357 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    permute_358 = rand_strided((96, 197, 64), (12608, 1, 197), device='cpu', dtype=torch.float32)
    permute_362 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_35 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_366 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_370 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_36 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_374 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_379 = rand_strided((96, 197, 197), (38809, 1, 197), device='cpu', dtype=torch.float32)
    permute_380 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    alias_23 = rand_strided((8, 12, 197, 197), (465708, 1, 2364, 12), device='cpu', dtype=torch.float32)
    permute_381 = rand_strided((96, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    permute_382 = rand_strided((96, 197, 64), (12608, 1, 197), device='cpu', dtype=torch.float32)
    permute_386 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_2, primals_3, primals_9, primals_10, primals_12, primals_13, primals_19, primals_20, primals_22, primals_23, primals_29, primals_30, primals_32, primals_33, primals_39, primals_40, primals_42, primals_43, primals_49, primals_50, primals_52, primals_53, primals_59, primals_60, primals_62, primals_63, primals_69, primals_70, primals_72, primals_73, primals_79, primals_80, primals_82, primals_83, primals_89, primals_90, primals_92, primals_93, primals_99, primals_100, primals_102, primals_103, primals_109, primals_110, primals_112, primals_113, primals_119, primals_120, primals_122, primals_124, primals_224, cat, getitem_1, rsqrt, view_1, view_4, view_13, addmm_1, mul_5, view_15, addmm_2, view_17, addmm_3, mul_11, view_19, view_22, view_31, addmm_5, mul_16, view_33, addmm_6, view_35, addmm_7, mul_22, view_37, view_40, view_49, addmm_9, mul_27, view_51, addmm_10, view_53, addmm_11, mul_33, view_55, view_58, view_67, addmm_13, mul_38, view_69, addmm_14, view_71, addmm_15, mul_44, view_73, view_76, view_85, addmm_17, mul_49, view_87, addmm_18, view_89, addmm_19, mul_55, view_91, view_94, view_103, addmm_21, mul_60, view_105, addmm_22, view_107, addmm_23, mul_66, view_109, view_112, view_121, addmm_25, mul_71, view_123, addmm_26, view_125, addmm_27, mul_77, view_127, view_130, view_139, addmm_29, mul_82, view_141, addmm_30, view_143, addmm_31, mul_88, view_145, view_148, view_157, addmm_33, mul_93, view_159, addmm_34, view_161, addmm_35, mul_99, view_163, view_166, view_175, addmm_37, mul_104, view_177, addmm_38, view_179, addmm_39, mul_110, view_181, view_184, view_193, addmm_41, mul_115, view_195, addmm_42, view_197, addmm_43, mul_121, view_199, view_202, view_211, addmm_45, mul_126, view_213, addmm_46, view_215, addmm_47, mul_132, clone_97, permute_98, div_12, permute_102, permute_106, div_14, permute_110, permute_115, permute_116, alias_12, permute_117, permute_118, permute_122, div_15, permute_126, permute_130, div_16, permute_134, permute_139, permute_140, alias_13, permute_141, permute_142, permute_146, div_17, permute_150, permute_154, div_18, permute_158, permute_163, permute_164, alias_14, permute_165, permute_166, permute_170, div_19, permute_174, permute_178, div_20, permute_182, permute_187, permute_188, alias_15, permute_189, permute_190, permute_194, div_21, permute_198, permute_202, div_22, permute_206, permute_211, permute_212, alias_16, permute_213, permute_214, permute_218, div_23, permute_222, permute_226, div_24, permute_230, permute_235, permute_236, alias_17, permute_237, permute_238, permute_242, div_25, permute_246, permute_250, div_26, permute_254, permute_259, permute_260, alias_18, permute_261, permute_262, permute_266, div_27, permute_270, permute_274, div_28, permute_278, permute_283, permute_284, alias_19, permute_285, permute_286, permute_290, div_29, permute_294, permute_298, div_30, permute_302, permute_307, permute_308, alias_20, permute_309, permute_310, permute_314, div_31, permute_318, permute_322, div_32, permute_326, permute_331, permute_332, alias_21, permute_333, permute_334, permute_338, div_33, permute_342, permute_346, div_34, permute_350, permute_355, permute_356, alias_22, permute_357, permute_358, permute_362, div_35, permute_366, permute_370, div_36, permute_374, permute_379, permute_380, alias_23, permute_381, permute_382, permute_386, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('beit_base_patch16_224', benchmark_compiled_module)
