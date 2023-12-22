
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


cpp_fused_native_layer_norm_backward_select_backward_sum_0 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                        {
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (768L*x0))];
                            auto tmp6 = in_ptr2[static_cast<long>(x2)];
                            auto tmp8 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (151296L*x0))];
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                            tmp_acc0 = tmp_acc0 + tmp7;
                            tmp_acc1 = tmp_acc1 + tmp9;
                        }
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = tmp_acc0;
                        out_ptr2[static_cast<long>(x1 + (197L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x1 + (197L*x0))];
                        auto tmp4 = in_ptr1[static_cast<long>(x2 + (768L*x0))];
                        auto tmp7 = in_ptr2[static_cast<long>(x2)];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp13 = in_ptr3[static_cast<long>(x2 + (768L*x1) + (151296L*x0))];
                        auto tmp14 = out_ptr2[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = c10::convert<int>(x1);
                        auto tmp2 = static_cast<int>(0);
                        auto tmp3 = tmp1 == tmp2;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp9 = static_cast<float>(768.0);
                        auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                        auto tmp12 = decltype(tmp10)(tmp10 - tmp11);
                        auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                        auto tmp16 = decltype(tmp12)(tmp12 - tmp15);
                        auto tmp17 = decltype(tmp0)(tmp0 * tmp16);
                        out_ptr3[static_cast<long>(x2 + (768L*x1) + (151296L*x0))] = tmp17;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp3 = in_ptr1[static_cast<long>(x0 + (768L*x1))];
                            auto tmp6 = in_ptr3[static_cast<long>(x0 + (768L*x2) + (151296L*x1))];
                            auto tmp0 = c10::convert<int>(x2);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            tmp_acc0 = tmp_acc0 + tmp7;
                            tmp_acc1 = tmp_acc1 + tmp5;
                        }
                    }
                    out_ptr4[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr5[static_cast<long>(x0)] = tmp_acc1;
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


cpp_fused_sum_3 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_4 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(16);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1210368L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (768L*x0) + (2304L*x2) + (453888L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_5 = async_compile.cpp('''
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


cpp_fused_gelu_gelu_backward_sum_6 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_backward_sum_7 = async_compile.cpp('''
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


cpp_fused_sum_8 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_9 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(16);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1210368L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (768L*x0) + (2304L*x2) + (453888L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_10 = async_compile.cpp('''
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


cpp_fused_gelu_gelu_backward_sum_11 = async_compile.cpp('''
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


cpp_fused_sum_13 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_14 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(16);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1210368L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (768L*x0) + (2304L*x2) + (453888L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_15 = async_compile.cpp('''
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


cpp_fused_gelu_gelu_backward_sum_16 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_backward_sum_17 = async_compile.cpp('''
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


cpp_fused_sum_18 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_19 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(16);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1210368L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (768L*x0) + (2304L*x2) + (453888L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_20 = async_compile.cpp('''
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


cpp_fused_gelu_gelu_backward_sum_21 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_backward_sum_22 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_24 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(16);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1210368L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (768L*x0) + (2304L*x2) + (453888L*x1)));
                        }
                    }
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


cpp_fused_gelu_gelu_backward_sum_26 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_backward_sum_27 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_29 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(16);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1210368L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (768L*x0) + (2304L*x2) + (453888L*x1)));
                        }
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


cpp_fused_gelu_gelu_backward_sum_31 = async_compile.cpp('''
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


cpp_fused_sum_33 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(16);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1210368L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (768L*x0) + (2304L*x2) + (453888L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_35 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_backward_sum_37 = async_compile.cpp('''
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


cpp_fused_sum_38 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_39 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(16);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1210368L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (768L*x0) + (2304L*x2) + (453888L*x1)));
                        }
                    }
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


cpp_fused_gelu_gelu_backward_sum_41 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_44 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(16);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1210368L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (768L*x0) + (2304L*x2) + (453888L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_45 = async_compile.cpp('''
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


cpp_fused_gelu_gelu_backward_sum_46 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_backward_sum_47 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_49 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(16);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1210368L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (768L*x0) + (2304L*x2) + (453888L*x1)));
                        }
                    }
                }
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


cpp_fused_add_native_layer_norm_backward_sum_52 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_54 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(16);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1210368L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (768L*x0) + (2304L*x2) + (453888L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_55 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_backward_sum_57 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_59 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(16);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-1210368L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x3 + (768L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (768L*x0) + (2304L*x2) + (453888L*x1)));
                        }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(151296L); x0+=static_cast<long>(8L))
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
                        tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
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
    primals_3, primals_5, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_47, primals_53, primals_59, primals_65, primals_71, primals_77, primals_83, primals_89, primals_95, primals_101, primals_107, primals_113, primals_119, primals_125, primals_131, primals_137, primals_143, primals_149, primals_153, mul, view_1, getitem_2, getitem_3, getitem_4, getitem_6, getitem_7, getitem_8, getitem_11, getitem_12, view_5, mul_2, view_7, addmm_2, view_9, mul_7, view_11, getitem_18, getitem_19, getitem_20, getitem_22, getitem_23, getitem_24, getitem_27, getitem_28, view_15, mul_9, view_17, addmm_6, view_19, mul_14, view_21, getitem_34, getitem_35, getitem_36, getitem_38, getitem_39, getitem_40, getitem_43, getitem_44, view_25, mul_16, view_27, addmm_10, view_29, mul_21, view_31, getitem_50, getitem_51, getitem_52, getitem_54, getitem_55, getitem_56, getitem_59, getitem_60, view_35, mul_23, view_37, addmm_14, view_39, mul_28, view_41, getitem_66, getitem_67, getitem_68, getitem_70, getitem_71, getitem_72, getitem_75, getitem_76, view_45, mul_30, view_47, addmm_18, view_49, mul_35, view_51, getitem_82, getitem_83, getitem_84, getitem_86, getitem_87, getitem_88, getitem_91, getitem_92, view_55, mul_37, view_57, addmm_22, view_59, mul_42, view_61, getitem_98, getitem_99, getitem_100, getitem_102, getitem_103, getitem_104, getitem_107, getitem_108, view_65, mul_44, view_67, addmm_26, view_69, mul_49, view_71, getitem_114, getitem_115, getitem_116, getitem_118, getitem_119, getitem_120, getitem_123, getitem_124, view_75, mul_51, view_77, addmm_30, view_79, mul_56, view_81, getitem_130, getitem_131, getitem_132, getitem_134, getitem_135, getitem_136, getitem_139, getitem_140, view_85, mul_58, view_87, addmm_34, view_89, mul_63, view_91, getitem_146, getitem_147, getitem_148, getitem_150, getitem_151, getitem_152, getitem_155, getitem_156, view_95, mul_65, view_97, addmm_38, view_99, mul_70, view_101, getitem_162, getitem_163, getitem_164, getitem_166, getitem_167, getitem_168, getitem_171, getitem_172, view_105, mul_72, view_107, addmm_42, view_109, mul_77, view_111, getitem_178, getitem_179, getitem_180, getitem_182, getitem_183, getitem_184, getitem_187, getitem_188, view_115, mul_79, view_117, addmm_46, view_119, mul_84, clone_37, permute_74, div, permute_78, permute_82, div_1, permute_86, alias_12, permute_92, div_2, permute_96, permute_100, div_3, permute_104, alias_13, permute_110, div_4, permute_114, permute_118, div_5, permute_122, alias_14, permute_128, div_6, permute_132, permute_136, div_7, permute_140, alias_15, permute_146, div_8, permute_150, permute_154, div_9, permute_158, alias_16, permute_164, div_10, permute_168, permute_172, div_11, permute_176, alias_17, permute_182, div_12, permute_186, permute_190, div_13, permute_194, alias_18, permute_200, div_14, permute_204, permute_208, div_15, permute_212, alias_19, permute_218, div_16, permute_222, permute_226, div_17, permute_230, alias_20, permute_236, div_18, permute_240, permute_244, div_19, permute_248, alias_21, permute_254, div_20, permute_258, permute_262, div_21, permute_266, alias_22, permute_272, div_22, permute_276, permute_280, div_23, permute_284, alias_23, permute_290, div_24, tangents_1 = args
    args.clear()
    assert_size_stride(primals_3, (768, 3, 16, 16), (768, 1, 48, 3))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_101, (768, ), (1, ))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_113, (768, ), (1, ))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_125, (768, ), (1, ))
    assert_size_stride(primals_131, (768, ), (1, ))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_149, (768, ), (1, ))
    assert_size_stride(primals_153, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(mul, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_1, (1576, 768), (768, 1))
    assert_size_stride(getitem_2, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_3, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_4, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_6, (8, 12, 197), (2364, 1, 12))
    assert_size_stride(getitem_7, (), ())
    assert_size_stride(getitem_8, (), ())
    assert_size_stride(getitem_11, (), ())
    assert_size_stride(getitem_12, (), ())
    assert_size_stride(view_5, (1576, 768), (768, 1))
    assert_size_stride(mul_2, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_7, (1576, 768), (768, 1))
    assert_size_stride(addmm_2, (1576, 3072), (3072, 1))
    assert_size_stride(view_9, (1576, 3072), (3072, 1))
    assert_size_stride(mul_7, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_11, (1576, 768), (768, 1))
    assert_size_stride(getitem_18, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_19, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_20, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_22, (8, 12, 197), (2364, 1, 12))
    assert_size_stride(getitem_23, (), ())
    assert_size_stride(getitem_24, (), ())
    assert_size_stride(getitem_27, (), ())
    assert_size_stride(getitem_28, (), ())
    assert_size_stride(view_15, (1576, 768), (768, 1))
    assert_size_stride(mul_9, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_17, (1576, 768), (768, 1))
    assert_size_stride(addmm_6, (1576, 3072), (3072, 1))
    assert_size_stride(view_19, (1576, 3072), (3072, 1))
    assert_size_stride(mul_14, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_21, (1576, 768), (768, 1))
    assert_size_stride(getitem_34, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_35, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_36, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_38, (8, 12, 197), (2364, 1, 12))
    assert_size_stride(getitem_39, (), ())
    assert_size_stride(getitem_40, (), ())
    assert_size_stride(getitem_43, (), ())
    assert_size_stride(getitem_44, (), ())
    assert_size_stride(view_25, (1576, 768), (768, 1))
    assert_size_stride(mul_16, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_27, (1576, 768), (768, 1))
    assert_size_stride(addmm_10, (1576, 3072), (3072, 1))
    assert_size_stride(view_29, (1576, 3072), (3072, 1))
    assert_size_stride(mul_21, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_31, (1576, 768), (768, 1))
    assert_size_stride(getitem_50, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_51, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_52, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_54, (8, 12, 197), (2364, 1, 12))
    assert_size_stride(getitem_55, (), ())
    assert_size_stride(getitem_56, (), ())
    assert_size_stride(getitem_59, (), ())
    assert_size_stride(getitem_60, (), ())
    assert_size_stride(view_35, (1576, 768), (768, 1))
    assert_size_stride(mul_23, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_37, (1576, 768), (768, 1))
    assert_size_stride(addmm_14, (1576, 3072), (3072, 1))
    assert_size_stride(view_39, (1576, 3072), (3072, 1))
    assert_size_stride(mul_28, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_41, (1576, 768), (768, 1))
    assert_size_stride(getitem_66, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_67, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_68, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_70, (8, 12, 197), (2364, 1, 12))
    assert_size_stride(getitem_71, (), ())
    assert_size_stride(getitem_72, (), ())
    assert_size_stride(getitem_75, (), ())
    assert_size_stride(getitem_76, (), ())
    assert_size_stride(view_45, (1576, 768), (768, 1))
    assert_size_stride(mul_30, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_47, (1576, 768), (768, 1))
    assert_size_stride(addmm_18, (1576, 3072), (3072, 1))
    assert_size_stride(view_49, (1576, 3072), (3072, 1))
    assert_size_stride(mul_35, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_51, (1576, 768), (768, 1))
    assert_size_stride(getitem_82, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_83, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_84, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_86, (8, 12, 197), (2364, 1, 12))
    assert_size_stride(getitem_87, (), ())
    assert_size_stride(getitem_88, (), ())
    assert_size_stride(getitem_91, (), ())
    assert_size_stride(getitem_92, (), ())
    assert_size_stride(view_55, (1576, 768), (768, 1))
    assert_size_stride(mul_37, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_57, (1576, 768), (768, 1))
    assert_size_stride(addmm_22, (1576, 3072), (3072, 1))
    assert_size_stride(view_59, (1576, 3072), (3072, 1))
    assert_size_stride(mul_42, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_61, (1576, 768), (768, 1))
    assert_size_stride(getitem_98, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_99, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_100, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_102, (8, 12, 197), (2364, 1, 12))
    assert_size_stride(getitem_103, (), ())
    assert_size_stride(getitem_104, (), ())
    assert_size_stride(getitem_107, (), ())
    assert_size_stride(getitem_108, (), ())
    assert_size_stride(view_65, (1576, 768), (768, 1))
    assert_size_stride(mul_44, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_67, (1576, 768), (768, 1))
    assert_size_stride(addmm_26, (1576, 3072), (3072, 1))
    assert_size_stride(view_69, (1576, 3072), (3072, 1))
    assert_size_stride(mul_49, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_71, (1576, 768), (768, 1))
    assert_size_stride(getitem_114, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_115, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_116, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_118, (8, 12, 197), (2364, 1, 12))
    assert_size_stride(getitem_119, (), ())
    assert_size_stride(getitem_120, (), ())
    assert_size_stride(getitem_123, (), ())
    assert_size_stride(getitem_124, (), ())
    assert_size_stride(view_75, (1576, 768), (768, 1))
    assert_size_stride(mul_51, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_77, (1576, 768), (768, 1))
    assert_size_stride(addmm_30, (1576, 3072), (3072, 1))
    assert_size_stride(view_79, (1576, 3072), (3072, 1))
    assert_size_stride(mul_56, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_81, (1576, 768), (768, 1))
    assert_size_stride(getitem_130, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_131, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_132, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_134, (8, 12, 197), (2364, 1, 12))
    assert_size_stride(getitem_135, (), ())
    assert_size_stride(getitem_136, (), ())
    assert_size_stride(getitem_139, (), ())
    assert_size_stride(getitem_140, (), ())
    assert_size_stride(view_85, (1576, 768), (768, 1))
    assert_size_stride(mul_58, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_87, (1576, 768), (768, 1))
    assert_size_stride(addmm_34, (1576, 3072), (3072, 1))
    assert_size_stride(view_89, (1576, 3072), (3072, 1))
    assert_size_stride(mul_63, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_91, (1576, 768), (768, 1))
    assert_size_stride(getitem_146, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_147, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_148, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_150, (8, 12, 197), (2364, 1, 12))
    assert_size_stride(getitem_151, (), ())
    assert_size_stride(getitem_152, (), ())
    assert_size_stride(getitem_155, (), ())
    assert_size_stride(getitem_156, (), ())
    assert_size_stride(view_95, (1576, 768), (768, 1))
    assert_size_stride(mul_65, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_97, (1576, 768), (768, 1))
    assert_size_stride(addmm_38, (1576, 3072), (3072, 1))
    assert_size_stride(view_99, (1576, 3072), (3072, 1))
    assert_size_stride(mul_70, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_101, (1576, 768), (768, 1))
    assert_size_stride(getitem_162, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_163, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_164, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_166, (8, 12, 197), (2364, 1, 12))
    assert_size_stride(getitem_167, (), ())
    assert_size_stride(getitem_168, (), ())
    assert_size_stride(getitem_171, (), ())
    assert_size_stride(getitem_172, (), ())
    assert_size_stride(view_105, (1576, 768), (768, 1))
    assert_size_stride(mul_72, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_107, (1576, 768), (768, 1))
    assert_size_stride(addmm_42, (1576, 3072), (3072, 1))
    assert_size_stride(view_109, (1576, 3072), (3072, 1))
    assert_size_stride(mul_77, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_111, (1576, 768), (768, 1))
    assert_size_stride(getitem_178, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_179, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_180, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_182, (8, 12, 197), (2364, 1, 12))
    assert_size_stride(getitem_183, (), ())
    assert_size_stride(getitem_184, (), ())
    assert_size_stride(getitem_187, (), ())
    assert_size_stride(getitem_188, (), ())
    assert_size_stride(view_115, (1576, 768), (768, 1))
    assert_size_stride(mul_79, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_117, (1576, 768), (768, 1))
    assert_size_stride(addmm_46, (1576, 3072), (3072, 1))
    assert_size_stride(view_119, (1576, 3072), (3072, 1))
    assert_size_stride(mul_84, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(clone_37, (8, 768), (768, 1))
    assert_size_stride(permute_74, (1000, 768), (768, 1))
    assert_size_stride(div, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_78, (768, 3072), (3072, 1))
    assert_size_stride(permute_82, (3072, 768), (768, 1))
    assert_size_stride(div_1, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_86, (768, 768), (768, 1))
    assert_size_stride(alias_12, (8, 12, 197, 64), (151296, 1, 768, 12))
    assert_size_stride(permute_92, (2304, 768), (768, 1))
    assert_size_stride(div_2, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_96, (768, 3072), (3072, 1))
    assert_size_stride(permute_100, (3072, 768), (768, 1))
    assert_size_stride(div_3, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_104, (768, 768), (768, 1))
    assert_size_stride(alias_13, (8, 12, 197, 64), (151296, 1, 768, 12))
    assert_size_stride(permute_110, (2304, 768), (768, 1))
    assert_size_stride(div_4, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_114, (768, 3072), (3072, 1))
    assert_size_stride(permute_118, (3072, 768), (768, 1))
    assert_size_stride(div_5, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_122, (768, 768), (768, 1))
    assert_size_stride(alias_14, (8, 12, 197, 64), (151296, 1, 768, 12))
    assert_size_stride(permute_128, (2304, 768), (768, 1))
    assert_size_stride(div_6, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_132, (768, 3072), (3072, 1))
    assert_size_stride(permute_136, (3072, 768), (768, 1))
    assert_size_stride(div_7, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_140, (768, 768), (768, 1))
    assert_size_stride(alias_15, (8, 12, 197, 64), (151296, 1, 768, 12))
    assert_size_stride(permute_146, (2304, 768), (768, 1))
    assert_size_stride(div_8, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_150, (768, 3072), (3072, 1))
    assert_size_stride(permute_154, (3072, 768), (768, 1))
    assert_size_stride(div_9, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_158, (768, 768), (768, 1))
    assert_size_stride(alias_16, (8, 12, 197, 64), (151296, 1, 768, 12))
    assert_size_stride(permute_164, (2304, 768), (768, 1))
    assert_size_stride(div_10, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_168, (768, 3072), (3072, 1))
    assert_size_stride(permute_172, (3072, 768), (768, 1))
    assert_size_stride(div_11, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_176, (768, 768), (768, 1))
    assert_size_stride(alias_17, (8, 12, 197, 64), (151296, 1, 768, 12))
    assert_size_stride(permute_182, (2304, 768), (768, 1))
    assert_size_stride(div_12, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_186, (768, 3072), (3072, 1))
    assert_size_stride(permute_190, (3072, 768), (768, 1))
    assert_size_stride(div_13, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_194, (768, 768), (768, 1))
    assert_size_stride(alias_18, (8, 12, 197, 64), (151296, 1, 768, 12))
    assert_size_stride(permute_200, (2304, 768), (768, 1))
    assert_size_stride(div_14, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_204, (768, 3072), (3072, 1))
    assert_size_stride(permute_208, (3072, 768), (768, 1))
    assert_size_stride(div_15, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_212, (768, 768), (768, 1))
    assert_size_stride(alias_19, (8, 12, 197, 64), (151296, 1, 768, 12))
    assert_size_stride(permute_218, (2304, 768), (768, 1))
    assert_size_stride(div_16, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_222, (768, 3072), (3072, 1))
    assert_size_stride(permute_226, (3072, 768), (768, 1))
    assert_size_stride(div_17, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_230, (768, 768), (768, 1))
    assert_size_stride(alias_20, (8, 12, 197, 64), (151296, 1, 768, 12))
    assert_size_stride(permute_236, (2304, 768), (768, 1))
    assert_size_stride(div_18, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_240, (768, 3072), (3072, 1))
    assert_size_stride(permute_244, (3072, 768), (768, 1))
    assert_size_stride(div_19, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_248, (768, 768), (768, 1))
    assert_size_stride(alias_21, (8, 12, 197, 64), (151296, 1, 768, 12))
    assert_size_stride(permute_254, (2304, 768), (768, 1))
    assert_size_stride(div_20, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_258, (768, 3072), (3072, 1))
    assert_size_stride(permute_262, (3072, 768), (768, 1))
    assert_size_stride(div_21, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_266, (768, 768), (768, 1))
    assert_size_stride(alias_22, (8, 12, 197, 64), (151296, 1, 768, 12))
    assert_size_stride(permute_272, (2304, 768), (768, 1))
    assert_size_stride(div_22, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_276, (768, 3072), (3072, 1))
    assert_size_stride(permute_280, (3072, 768), (768, 1))
    assert_size_stride(div_23, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_284, (768, 768), (768, 1))
    assert_size_stride(alias_23, (8, 12, 197, 64), (151296, 1, 768, 12))
    assert_size_stride(permute_290, (2304, 768), (768, 1))
    assert_size_stride(div_24, (8, 197, 1), (197, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_74, out=buf0)
    del permute_74
    buf1 = empty((1000, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_37, out=buf1)
    del clone_37
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf5 = empty((8, 197, 768), device='cpu', dtype=torch.float32)
    buf6 = empty((768, ), device='cpu', dtype=torch.float32)
    buf7 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_backward_select_backward_sum_0(c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(mul_84.data_ptr()), c_void_p(div.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del buf0
    del div
    del mul_84
    del primals_149
    del tangents_1
    buf8 = empty((1576, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (1576, 768), (768, 1), 0), permute_78, out=buf8)
    del permute_78
    buf9 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (768, 1576), (1, 768), 0), view_119, out=buf9)
    del view_119
    buf10 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf11 = reinterpret_tensor(buf8, (8, 197, 3072), (605184, 3072, 1), 0); del buf8  # reuse
    cpp_fused_gelu_gelu_backward_sum_1(c_void_p(buf11.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(addmm_46.data_ptr()), c_void_p(buf10.data_ptr()))
    del addmm_46
    buf12 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (1576, 3072), (3072, 1), 0), permute_82, out=buf12)
    del permute_82
    buf13 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (3072, 1576), (1, 3072), 0), view_117, out=buf13)
    del view_117
    buf14 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf15 = buf4; del buf4  # reuse
    buf16 = buf3; del buf3  # reuse
    buf17 = empty((768, ), device='cpu', dtype=torch.float32)
    buf18 = empty((768, ), device='cpu', dtype=torch.float32)
    buf19 = reinterpret_tensor(buf12, (8, 197, 768), (151296, 768, 1), 0); del buf12  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_2(c_void_p(buf19.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(mul_79.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(div_1.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()))
    del div_1
    del mul_79
    del primals_143
    buf20 = reinterpret_tensor(buf5, (1576, 768), (768, 1), 0); del buf5  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf19, (1576, 768), (768, 1), 0), permute_86, out=buf20)
    del permute_86
    buf21 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf19, (768, 1576), (1, 768), 0), view_115, out=buf21)
    del view_115
    buf22 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_3(c_void_p(buf19.data_ptr()), c_void_p(buf22.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf23 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf20, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_178, getitem_179, getitem_180, alias_12, getitem_182, getitem_183, getitem_184, 0, 0, 0.0, False, getitem_187, getitem_188)
    del alias_12
    del buf20
    del getitem_178
    del getitem_179
    del getitem_180
    del getitem_182
    del getitem_183
    del getitem_184
    del getitem_187
    del getitem_188
    buf24 = buf23[0]
    buf25 = buf23[1]
    buf26 = buf23[2]
    del buf23
    buf27 = empty((8, 197, 3, 12, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_4(c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()))
    del buf24
    del buf25
    buf28 = reinterpret_tensor(buf26, (1576, 768), (768, 1), 0); del buf26  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf27, (1576, 2304), (2304, 1), 0), permute_92, out=buf28)
    del permute_92
    buf29 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf27, (2304, 1576), (1, 2304), 0), view_111, out=buf29)
    del view_111
    buf30 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf31 = buf16; del buf16  # reuse
    buf32 = buf15; del buf15  # reuse
    buf33 = empty((768, ), device='cpu', dtype=torch.float32)
    buf34 = empty((768, ), device='cpu', dtype=torch.float32)
    buf35 = buf19; del buf19  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_5(c_void_p(buf35.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(mul_77.data_ptr()), c_void_p(div_2.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()))
    del div_2
    del mul_77
    del primals_137
    buf36 = reinterpret_tensor(buf11, (1576, 3072), (3072, 1), 0); del buf11  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf35, (1576, 768), (768, 1), 0), permute_96, out=buf36)
    del permute_96
    buf37 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf35, (768, 1576), (1, 768), 0), view_109, out=buf37)
    del view_109
    buf38 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf39 = reinterpret_tensor(buf36, (8, 197, 3072), (605184, 3072, 1), 0); del buf36  # reuse
    cpp_fused_gelu_gelu_backward_sum_6(c_void_p(buf39.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(addmm_42.data_ptr()), c_void_p(buf38.data_ptr()))
    del addmm_42
    buf40 = buf28; del buf28  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf39, (1576, 3072), (3072, 1), 0), permute_100, out=buf40)
    del permute_100
    buf41 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf39, (3072, 1576), (1, 3072), 0), view_107, out=buf41)
    del view_107
    buf42 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf43 = buf32; del buf32  # reuse
    buf44 = buf31; del buf31  # reuse
    buf45 = empty((768, ), device='cpu', dtype=torch.float32)
    buf46 = empty((768, ), device='cpu', dtype=torch.float32)
    buf47 = buf35; del buf35  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_7(c_void_p(buf47.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(mul_72.data_ptr()), c_void_p(div_3.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()))
    del div_3
    del mul_72
    del primals_131
    buf48 = buf40; del buf40  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf47, (1576, 768), (768, 1), 0), permute_104, out=buf48)
    del permute_104
    buf49 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf47, (768, 1576), (1, 768), 0), view_105, out=buf49)
    del view_105
    buf50 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_8(c_void_p(buf47.data_ptr()), c_void_p(buf50.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf51 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf48, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_162, getitem_163, getitem_164, alias_13, getitem_166, getitem_167, getitem_168, 0, 0, 0.0, False, getitem_171, getitem_172)
    del alias_13
    del buf48
    del getitem_162
    del getitem_163
    del getitem_164
    del getitem_166
    del getitem_167
    del getitem_168
    del getitem_171
    del getitem_172
    buf52 = buf51[0]
    buf53 = buf51[1]
    buf54 = buf51[2]
    del buf51
    buf55 = buf27; del buf27  # reuse
    cpp_fused_clone_9(c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()))
    del buf52
    del buf53
    buf56 = reinterpret_tensor(buf54, (1576, 768), (768, 1), 0); del buf54  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf55, (1576, 2304), (2304, 1), 0), permute_110, out=buf56)
    del permute_110
    buf57 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf55, (2304, 1576), (1, 2304), 0), view_101, out=buf57)
    del view_101
    buf58 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf59 = buf44; del buf44  # reuse
    buf60 = buf43; del buf43  # reuse
    buf61 = empty((768, ), device='cpu', dtype=torch.float32)
    buf62 = empty((768, ), device='cpu', dtype=torch.float32)
    buf63 = buf47; del buf47  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_10(c_void_p(buf63.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(mul_70.data_ptr()), c_void_p(div_4.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    del div_4
    del mul_70
    del primals_125
    buf64 = reinterpret_tensor(buf39, (1576, 3072), (3072, 1), 0); del buf39  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf63, (1576, 768), (768, 1), 0), permute_114, out=buf64)
    del permute_114
    buf65 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf63, (768, 1576), (1, 768), 0), view_99, out=buf65)
    del view_99
    buf66 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf67 = reinterpret_tensor(buf64, (8, 197, 3072), (605184, 3072, 1), 0); del buf64  # reuse
    cpp_fused_gelu_gelu_backward_sum_11(c_void_p(buf67.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(addmm_38.data_ptr()), c_void_p(buf66.data_ptr()))
    del addmm_38
    buf68 = buf56; del buf56  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf67, (1576, 3072), (3072, 1), 0), permute_118, out=buf68)
    del permute_118
    buf69 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf67, (3072, 1576), (1, 3072), 0), view_97, out=buf69)
    del view_97
    buf70 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf71 = buf60; del buf60  # reuse
    buf72 = buf59; del buf59  # reuse
    buf73 = empty((768, ), device='cpu', dtype=torch.float32)
    buf74 = empty((768, ), device='cpu', dtype=torch.float32)
    buf75 = buf63; del buf63  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_12(c_void_p(buf75.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(mul_65.data_ptr()), c_void_p(div_5.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()))
    del div_5
    del mul_65
    del primals_119
    buf76 = buf68; del buf68  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf75, (1576, 768), (768, 1), 0), permute_122, out=buf76)
    del permute_122
    buf77 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf75, (768, 1576), (1, 768), 0), view_95, out=buf77)
    del view_95
    buf78 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_13(c_void_p(buf75.data_ptr()), c_void_p(buf78.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf79 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf76, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_146, getitem_147, getitem_148, alias_14, getitem_150, getitem_151, getitem_152, 0, 0, 0.0, False, getitem_155, getitem_156)
    del alias_14
    del buf76
    del getitem_146
    del getitem_147
    del getitem_148
    del getitem_150
    del getitem_151
    del getitem_152
    del getitem_155
    del getitem_156
    buf80 = buf79[0]
    buf81 = buf79[1]
    buf82 = buf79[2]
    del buf79
    buf83 = buf55; del buf55  # reuse
    cpp_fused_clone_14(c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()))
    del buf80
    del buf81
    buf84 = reinterpret_tensor(buf82, (1576, 768), (768, 1), 0); del buf82  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf83, (1576, 2304), (2304, 1), 0), permute_128, out=buf84)
    del permute_128
    buf85 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf83, (2304, 1576), (1, 2304), 0), view_91, out=buf85)
    del view_91
    buf86 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf87 = buf72; del buf72  # reuse
    buf88 = buf71; del buf71  # reuse
    buf89 = empty((768, ), device='cpu', dtype=torch.float32)
    buf90 = empty((768, ), device='cpu', dtype=torch.float32)
    buf91 = buf75; del buf75  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_15(c_void_p(buf91.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(mul_63.data_ptr()), c_void_p(div_6.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()))
    del div_6
    del mul_63
    del primals_113
    buf92 = reinterpret_tensor(buf67, (1576, 3072), (3072, 1), 0); del buf67  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf91, (1576, 768), (768, 1), 0), permute_132, out=buf92)
    del permute_132
    buf93 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf91, (768, 1576), (1, 768), 0), view_89, out=buf93)
    del view_89
    buf94 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf95 = reinterpret_tensor(buf92, (8, 197, 3072), (605184, 3072, 1), 0); del buf92  # reuse
    cpp_fused_gelu_gelu_backward_sum_16(c_void_p(buf95.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(buf94.data_ptr()))
    del addmm_34
    buf96 = buf84; del buf84  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf95, (1576, 3072), (3072, 1), 0), permute_136, out=buf96)
    del permute_136
    buf97 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf95, (3072, 1576), (1, 3072), 0), view_87, out=buf97)
    del view_87
    buf98 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf99 = buf88; del buf88  # reuse
    buf100 = buf87; del buf87  # reuse
    buf101 = empty((768, ), device='cpu', dtype=torch.float32)
    buf102 = empty((768, ), device='cpu', dtype=torch.float32)
    buf103 = buf91; del buf91  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_17(c_void_p(buf103.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(mul_58.data_ptr()), c_void_p(div_7.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()))
    del div_7
    del mul_58
    del primals_107
    buf104 = buf96; del buf96  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf103, (1576, 768), (768, 1), 0), permute_140, out=buf104)
    del permute_140
    buf105 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf103, (768, 1576), (1, 768), 0), view_85, out=buf105)
    del view_85
    buf106 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_18(c_void_p(buf103.data_ptr()), c_void_p(buf106.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf107 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf104, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_130, getitem_131, getitem_132, alias_15, getitem_134, getitem_135, getitem_136, 0, 0, 0.0, False, getitem_139, getitem_140)
    del alias_15
    del buf104
    del getitem_130
    del getitem_131
    del getitem_132
    del getitem_134
    del getitem_135
    del getitem_136
    del getitem_139
    del getitem_140
    buf108 = buf107[0]
    buf109 = buf107[1]
    buf110 = buf107[2]
    del buf107
    buf111 = buf83; del buf83  # reuse
    cpp_fused_clone_19(c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()))
    del buf108
    del buf109
    buf112 = reinterpret_tensor(buf110, (1576, 768), (768, 1), 0); del buf110  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf111, (1576, 2304), (2304, 1), 0), permute_146, out=buf112)
    del permute_146
    buf113 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf111, (2304, 1576), (1, 2304), 0), view_81, out=buf113)
    del view_81
    buf114 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf115 = buf99; del buf99  # reuse
    buf116 = buf100; del buf100  # reuse
    buf117 = empty((768, ), device='cpu', dtype=torch.float32)
    buf118 = empty((768, ), device='cpu', dtype=torch.float32)
    buf119 = buf103; del buf103  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_20(c_void_p(buf119.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(mul_56.data_ptr()), c_void_p(div_8.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()))
    del div_8
    del mul_56
    del primals_101
    buf120 = reinterpret_tensor(buf95, (1576, 3072), (3072, 1), 0); del buf95  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf119, (1576, 768), (768, 1), 0), permute_150, out=buf120)
    del permute_150
    buf121 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf119, (768, 1576), (1, 768), 0), view_79, out=buf121)
    del view_79
    buf122 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf123 = reinterpret_tensor(buf120, (8, 197, 3072), (605184, 3072, 1), 0); del buf120  # reuse
    cpp_fused_gelu_gelu_backward_sum_21(c_void_p(buf123.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(addmm_30.data_ptr()), c_void_p(buf122.data_ptr()))
    del addmm_30
    buf124 = buf112; del buf112  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf123, (1576, 3072), (3072, 1), 0), permute_154, out=buf124)
    del permute_154
    buf125 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf123, (3072, 1576), (1, 3072), 0), view_77, out=buf125)
    del view_77
    buf126 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf127 = buf116; del buf116  # reuse
    buf128 = buf115; del buf115  # reuse
    buf129 = empty((768, ), device='cpu', dtype=torch.float32)
    buf130 = empty((768, ), device='cpu', dtype=torch.float32)
    buf131 = buf119; del buf119  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_22(c_void_p(buf131.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(mul_51.data_ptr()), c_void_p(div_9.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()))
    del div_9
    del mul_51
    del primals_95
    buf132 = buf124; del buf124  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf131, (1576, 768), (768, 1), 0), permute_158, out=buf132)
    del permute_158
    buf133 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf131, (768, 1576), (1, 768), 0), view_75, out=buf133)
    del view_75
    buf134 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_23(c_void_p(buf131.data_ptr()), c_void_p(buf134.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf135 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf132, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_114, getitem_115, getitem_116, alias_16, getitem_118, getitem_119, getitem_120, 0, 0, 0.0, False, getitem_123, getitem_124)
    del alias_16
    del buf132
    del getitem_114
    del getitem_115
    del getitem_116
    del getitem_118
    del getitem_119
    del getitem_120
    del getitem_123
    del getitem_124
    buf136 = buf135[0]
    buf137 = buf135[1]
    buf138 = buf135[2]
    del buf135
    buf139 = buf111; del buf111  # reuse
    cpp_fused_clone_24(c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()))
    del buf136
    del buf137
    buf140 = reinterpret_tensor(buf138, (1576, 768), (768, 1), 0); del buf138  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf139, (1576, 2304), (2304, 1), 0), permute_164, out=buf140)
    del permute_164
    buf141 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf139, (2304, 1576), (1, 2304), 0), view_71, out=buf141)
    del view_71
    buf142 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf143 = buf128; del buf128  # reuse
    buf144 = buf127; del buf127  # reuse
    buf145 = empty((768, ), device='cpu', dtype=torch.float32)
    buf146 = empty((768, ), device='cpu', dtype=torch.float32)
    buf147 = buf131; del buf131  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_25(c_void_p(buf147.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(mul_49.data_ptr()), c_void_p(div_10.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()))
    del div_10
    del mul_49
    del primals_89
    buf148 = reinterpret_tensor(buf123, (1576, 3072), (3072, 1), 0); del buf123  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf147, (1576, 768), (768, 1), 0), permute_168, out=buf148)
    del permute_168
    buf149 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf147, (768, 1576), (1, 768), 0), view_69, out=buf149)
    del view_69
    buf150 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf151 = reinterpret_tensor(buf148, (8, 197, 3072), (605184, 3072, 1), 0); del buf148  # reuse
    cpp_fused_gelu_gelu_backward_sum_26(c_void_p(buf151.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(addmm_26.data_ptr()), c_void_p(buf150.data_ptr()))
    del addmm_26
    buf152 = buf140; del buf140  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf151, (1576, 3072), (3072, 1), 0), permute_172, out=buf152)
    del permute_172
    buf153 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf151, (3072, 1576), (1, 3072), 0), view_67, out=buf153)
    del view_67
    buf154 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf155 = buf144; del buf144  # reuse
    buf156 = buf143; del buf143  # reuse
    buf157 = empty((768, ), device='cpu', dtype=torch.float32)
    buf158 = empty((768, ), device='cpu', dtype=torch.float32)
    buf159 = buf147; del buf147  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_27(c_void_p(buf159.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(mul_44.data_ptr()), c_void_p(div_11.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()))
    del div_11
    del mul_44
    del primals_83
    buf160 = buf152; del buf152  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf159, (1576, 768), (768, 1), 0), permute_176, out=buf160)
    del permute_176
    buf161 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf159, (768, 1576), (1, 768), 0), view_65, out=buf161)
    del view_65
    buf162 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_28(c_void_p(buf159.data_ptr()), c_void_p(buf162.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf163 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf160, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_98, getitem_99, getitem_100, alias_17, getitem_102, getitem_103, getitem_104, 0, 0, 0.0, False, getitem_107, getitem_108)
    del alias_17
    del buf160
    del getitem_100
    del getitem_102
    del getitem_103
    del getitem_104
    del getitem_107
    del getitem_108
    del getitem_98
    del getitem_99
    buf164 = buf163[0]
    buf165 = buf163[1]
    buf166 = buf163[2]
    del buf163
    buf167 = buf139; del buf139  # reuse
    cpp_fused_clone_29(c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()))
    del buf164
    del buf165
    buf168 = reinterpret_tensor(buf166, (1576, 768), (768, 1), 0); del buf166  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf167, (1576, 2304), (2304, 1), 0), permute_182, out=buf168)
    del permute_182
    buf169 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf167, (2304, 1576), (1, 2304), 0), view_61, out=buf169)
    del view_61
    buf170 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf171 = buf156; del buf156  # reuse
    buf172 = buf155; del buf155  # reuse
    buf173 = empty((768, ), device='cpu', dtype=torch.float32)
    buf174 = empty((768, ), device='cpu', dtype=torch.float32)
    buf175 = buf159; del buf159  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_30(c_void_p(buf175.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(mul_42.data_ptr()), c_void_p(div_12.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    del div_12
    del mul_42
    del primals_77
    buf176 = reinterpret_tensor(buf151, (1576, 3072), (3072, 1), 0); del buf151  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf175, (1576, 768), (768, 1), 0), permute_186, out=buf176)
    del permute_186
    buf177 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf175, (768, 1576), (1, 768), 0), view_59, out=buf177)
    del view_59
    buf178 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf179 = reinterpret_tensor(buf176, (8, 197, 3072), (605184, 3072, 1), 0); del buf176  # reuse
    cpp_fused_gelu_gelu_backward_sum_31(c_void_p(buf179.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(buf178.data_ptr()))
    del addmm_22
    buf180 = buf168; del buf168  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf179, (1576, 3072), (3072, 1), 0), permute_190, out=buf180)
    del permute_190
    buf181 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf179, (3072, 1576), (1, 3072), 0), view_57, out=buf181)
    del view_57
    buf182 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf183 = buf172; del buf172  # reuse
    buf184 = buf171; del buf171  # reuse
    buf185 = empty((768, ), device='cpu', dtype=torch.float32)
    buf186 = empty((768, ), device='cpu', dtype=torch.float32)
    buf187 = buf175; del buf175  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_32(c_void_p(buf187.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(mul_37.data_ptr()), c_void_p(div_13.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()))
    del div_13
    del mul_37
    del primals_71
    buf188 = buf180; del buf180  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf187, (1576, 768), (768, 1), 0), permute_194, out=buf188)
    del permute_194
    buf189 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf187, (768, 1576), (1, 768), 0), view_55, out=buf189)
    del view_55
    buf190 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_33(c_void_p(buf187.data_ptr()), c_void_p(buf190.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf191 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf188, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_82, getitem_83, getitem_84, alias_18, getitem_86, getitem_87, getitem_88, 0, 0, 0.0, False, getitem_91, getitem_92)
    del alias_18
    del buf188
    del getitem_82
    del getitem_83
    del getitem_84
    del getitem_86
    del getitem_87
    del getitem_88
    del getitem_91
    del getitem_92
    buf192 = buf191[0]
    buf193 = buf191[1]
    buf194 = buf191[2]
    del buf191
    buf195 = buf167; del buf167  # reuse
    cpp_fused_clone_34(c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()))
    del buf192
    del buf193
    buf196 = reinterpret_tensor(buf194, (1576, 768), (768, 1), 0); del buf194  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf195, (1576, 2304), (2304, 1), 0), permute_200, out=buf196)
    del permute_200
    buf197 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf195, (2304, 1576), (1, 2304), 0), view_51, out=buf197)
    del view_51
    buf198 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf199 = buf184; del buf184  # reuse
    buf200 = buf183; del buf183  # reuse
    buf201 = empty((768, ), device='cpu', dtype=torch.float32)
    buf202 = empty((768, ), device='cpu', dtype=torch.float32)
    buf203 = buf187; del buf187  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_35(c_void_p(buf203.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(mul_35.data_ptr()), c_void_p(div_14.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()))
    del div_14
    del mul_35
    del primals_65
    buf204 = reinterpret_tensor(buf179, (1576, 3072), (3072, 1), 0); del buf179  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf203, (1576, 768), (768, 1), 0), permute_204, out=buf204)
    del permute_204
    buf205 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf203, (768, 1576), (1, 768), 0), view_49, out=buf205)
    del view_49
    buf206 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf207 = reinterpret_tensor(buf204, (8, 197, 3072), (605184, 3072, 1), 0); del buf204  # reuse
    cpp_fused_gelu_gelu_backward_sum_36(c_void_p(buf207.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(addmm_18.data_ptr()), c_void_p(buf206.data_ptr()))
    del addmm_18
    buf208 = buf196; del buf196  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf207, (1576, 3072), (3072, 1), 0), permute_208, out=buf208)
    del permute_208
    buf209 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf207, (3072, 1576), (1, 3072), 0), view_47, out=buf209)
    del view_47
    buf210 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf211 = buf200; del buf200  # reuse
    buf212 = buf199; del buf199  # reuse
    buf213 = empty((768, ), device='cpu', dtype=torch.float32)
    buf214 = empty((768, ), device='cpu', dtype=torch.float32)
    buf215 = buf203; del buf203  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_37(c_void_p(buf215.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(mul_30.data_ptr()), c_void_p(div_15.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()))
    del div_15
    del mul_30
    del primals_59
    buf216 = buf208; del buf208  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf215, (1576, 768), (768, 1), 0), permute_212, out=buf216)
    del permute_212
    buf217 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf215, (768, 1576), (1, 768), 0), view_45, out=buf217)
    del view_45
    buf218 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_38(c_void_p(buf215.data_ptr()), c_void_p(buf218.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf219 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf216, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_66, getitem_67, getitem_68, alias_19, getitem_70, getitem_71, getitem_72, 0, 0, 0.0, False, getitem_75, getitem_76)
    del alias_19
    del buf216
    del getitem_66
    del getitem_67
    del getitem_68
    del getitem_70
    del getitem_71
    del getitem_72
    del getitem_75
    del getitem_76
    buf220 = buf219[0]
    buf221 = buf219[1]
    buf222 = buf219[2]
    del buf219
    buf223 = buf195; del buf195  # reuse
    cpp_fused_clone_39(c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()))
    del buf220
    del buf221
    buf224 = reinterpret_tensor(buf222, (1576, 768), (768, 1), 0); del buf222  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf223, (1576, 2304), (2304, 1), 0), permute_218, out=buf224)
    del permute_218
    buf225 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf223, (2304, 1576), (1, 2304), 0), view_41, out=buf225)
    del view_41
    buf226 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf227 = buf212; del buf212  # reuse
    buf228 = buf211; del buf211  # reuse
    buf229 = empty((768, ), device='cpu', dtype=torch.float32)
    buf230 = empty((768, ), device='cpu', dtype=torch.float32)
    buf231 = buf215; del buf215  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_40(c_void_p(buf231.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(mul_28.data_ptr()), c_void_p(div_16.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()))
    del div_16
    del mul_28
    del primals_53
    buf232 = reinterpret_tensor(buf207, (1576, 3072), (3072, 1), 0); del buf207  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf231, (1576, 768), (768, 1), 0), permute_222, out=buf232)
    del permute_222
    buf233 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf231, (768, 1576), (1, 768), 0), view_39, out=buf233)
    del view_39
    buf234 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf235 = reinterpret_tensor(buf232, (8, 197, 3072), (605184, 3072, 1), 0); del buf232  # reuse
    cpp_fused_gelu_gelu_backward_sum_41(c_void_p(buf235.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(addmm_14.data_ptr()), c_void_p(buf234.data_ptr()))
    del addmm_14
    buf236 = buf224; del buf224  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf235, (1576, 3072), (3072, 1), 0), permute_226, out=buf236)
    del permute_226
    buf237 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf235, (3072, 1576), (1, 3072), 0), view_37, out=buf237)
    del view_37
    buf238 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf239 = buf228; del buf228  # reuse
    buf240 = buf227; del buf227  # reuse
    buf241 = empty((768, ), device='cpu', dtype=torch.float32)
    buf242 = empty((768, ), device='cpu', dtype=torch.float32)
    buf243 = buf231; del buf231  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_42(c_void_p(buf243.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(mul_23.data_ptr()), c_void_p(div_17.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()))
    del div_17
    del mul_23
    del primals_47
    buf244 = buf236; del buf236  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf243, (1576, 768), (768, 1), 0), permute_230, out=buf244)
    del permute_230
    buf245 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf243, (768, 1576), (1, 768), 0), view_35, out=buf245)
    del view_35
    buf246 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_43(c_void_p(buf243.data_ptr()), c_void_p(buf246.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf247 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf244, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_50, getitem_51, getitem_52, alias_20, getitem_54, getitem_55, getitem_56, 0, 0, 0.0, False, getitem_59, getitem_60)
    del alias_20
    del buf244
    del getitem_50
    del getitem_51
    del getitem_52
    del getitem_54
    del getitem_55
    del getitem_56
    del getitem_59
    del getitem_60
    buf248 = buf247[0]
    buf249 = buf247[1]
    buf250 = buf247[2]
    del buf247
    buf251 = buf223; del buf223  # reuse
    cpp_fused_clone_44(c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()))
    del buf248
    del buf249
    buf252 = reinterpret_tensor(buf250, (1576, 768), (768, 1), 0); del buf250  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf251, (1576, 2304), (2304, 1), 0), permute_236, out=buf252)
    del permute_236
    buf253 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf251, (2304, 1576), (1, 2304), 0), view_31, out=buf253)
    del view_31
    buf254 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf255 = buf240; del buf240  # reuse
    buf256 = buf239; del buf239  # reuse
    buf257 = empty((768, ), device='cpu', dtype=torch.float32)
    buf258 = empty((768, ), device='cpu', dtype=torch.float32)
    buf259 = buf243; del buf243  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_45(c_void_p(buf259.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(mul_21.data_ptr()), c_void_p(div_18.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()))
    del div_18
    del mul_21
    del primals_41
    buf260 = reinterpret_tensor(buf235, (1576, 3072), (3072, 1), 0); del buf235  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf259, (1576, 768), (768, 1), 0), permute_240, out=buf260)
    del permute_240
    buf261 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf259, (768, 1576), (1, 768), 0), view_29, out=buf261)
    del view_29
    buf262 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf263 = reinterpret_tensor(buf260, (8, 197, 3072), (605184, 3072, 1), 0); del buf260  # reuse
    cpp_fused_gelu_gelu_backward_sum_46(c_void_p(buf263.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(buf262.data_ptr()))
    del addmm_10
    buf264 = buf252; del buf252  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf263, (1576, 3072), (3072, 1), 0), permute_244, out=buf264)
    del permute_244
    buf265 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf263, (3072, 1576), (1, 3072), 0), view_27, out=buf265)
    del view_27
    buf266 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf267 = buf256; del buf256  # reuse
    buf268 = buf255; del buf255  # reuse
    buf269 = empty((768, ), device='cpu', dtype=torch.float32)
    buf270 = empty((768, ), device='cpu', dtype=torch.float32)
    buf271 = buf259; del buf259  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_47(c_void_p(buf271.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(mul_16.data_ptr()), c_void_p(div_19.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()))
    del div_19
    del mul_16
    del primals_35
    buf272 = buf264; del buf264  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf271, (1576, 768), (768, 1), 0), permute_248, out=buf272)
    del permute_248
    buf273 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf271, (768, 1576), (1, 768), 0), view_25, out=buf273)
    del view_25
    buf274 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_48(c_void_p(buf271.data_ptr()), c_void_p(buf274.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf275 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf272, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_34, getitem_35, getitem_36, alias_21, getitem_38, getitem_39, getitem_40, 0, 0, 0.0, False, getitem_43, getitem_44)
    del alias_21
    del buf272
    del getitem_34
    del getitem_35
    del getitem_36
    del getitem_38
    del getitem_39
    del getitem_40
    del getitem_43
    del getitem_44
    buf276 = buf275[0]
    buf277 = buf275[1]
    buf278 = buf275[2]
    del buf275
    buf279 = buf251; del buf251  # reuse
    cpp_fused_clone_49(c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()))
    del buf276
    del buf277
    buf280 = reinterpret_tensor(buf278, (1576, 768), (768, 1), 0); del buf278  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf279, (1576, 2304), (2304, 1), 0), permute_254, out=buf280)
    del permute_254
    buf281 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf279, (2304, 1576), (1, 2304), 0), view_21, out=buf281)
    del view_21
    buf282 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf283 = buf268; del buf268  # reuse
    buf284 = buf267; del buf267  # reuse
    buf285 = empty((768, ), device='cpu', dtype=torch.float32)
    buf286 = empty((768, ), device='cpu', dtype=torch.float32)
    buf287 = buf271; del buf271  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_50(c_void_p(buf287.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(mul_14.data_ptr()), c_void_p(div_20.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()))
    del div_20
    del mul_14
    del primals_29
    buf288 = reinterpret_tensor(buf263, (1576, 3072), (3072, 1), 0); del buf263  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf287, (1576, 768), (768, 1), 0), permute_258, out=buf288)
    del permute_258
    buf289 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf287, (768, 1576), (1, 768), 0), view_19, out=buf289)
    del view_19
    buf290 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf291 = reinterpret_tensor(buf288, (8, 197, 3072), (605184, 3072, 1), 0); del buf288  # reuse
    cpp_fused_gelu_gelu_backward_sum_51(c_void_p(buf291.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(addmm_6.data_ptr()), c_void_p(buf290.data_ptr()))
    del addmm_6
    buf292 = buf280; del buf280  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf291, (1576, 3072), (3072, 1), 0), permute_262, out=buf292)
    del permute_262
    buf293 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf291, (3072, 1576), (1, 3072), 0), view_17, out=buf293)
    del view_17
    buf294 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf295 = buf284; del buf284  # reuse
    buf296 = buf283; del buf283  # reuse
    buf297 = empty((768, ), device='cpu', dtype=torch.float32)
    buf298 = empty((768, ), device='cpu', dtype=torch.float32)
    buf299 = buf287; del buf287  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_52(c_void_p(buf299.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(mul_9.data_ptr()), c_void_p(div_21.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()))
    del div_21
    del mul_9
    del primals_23
    buf300 = buf292; del buf292  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf299, (1576, 768), (768, 1), 0), permute_266, out=buf300)
    del permute_266
    buf301 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf299, (768, 1576), (1, 768), 0), view_15, out=buf301)
    del view_15
    buf302 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_53(c_void_p(buf299.data_ptr()), c_void_p(buf302.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf303 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf300, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_18, getitem_19, getitem_20, alias_22, getitem_22, getitem_23, getitem_24, 0, 0, 0.0, False, getitem_27, getitem_28)
    del alias_22
    del buf300
    del getitem_18
    del getitem_19
    del getitem_20
    del getitem_22
    del getitem_23
    del getitem_24
    del getitem_27
    del getitem_28
    buf304 = buf303[0]
    buf305 = buf303[1]
    buf306 = buf303[2]
    del buf303
    buf307 = buf279; del buf279  # reuse
    cpp_fused_clone_54(c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()))
    del buf304
    del buf305
    buf308 = reinterpret_tensor(buf306, (1576, 768), (768, 1), 0); del buf306  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf307, (1576, 2304), (2304, 1), 0), permute_272, out=buf308)
    del permute_272
    buf309 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf307, (2304, 1576), (1, 2304), 0), view_11, out=buf309)
    del view_11
    buf310 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf311 = buf296; del buf296  # reuse
    buf312 = buf295; del buf295  # reuse
    buf313 = empty((768, ), device='cpu', dtype=torch.float32)
    buf314 = empty((768, ), device='cpu', dtype=torch.float32)
    buf315 = buf299; del buf299  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_55(c_void_p(buf315.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(mul_7.data_ptr()), c_void_p(div_22.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()))
    del div_22
    del mul_7
    del primals_17
    buf316 = reinterpret_tensor(buf291, (1576, 3072), (3072, 1), 0); del buf291  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf315, (1576, 768), (768, 1), 0), permute_276, out=buf316)
    del permute_276
    buf317 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf315, (768, 1576), (1, 768), 0), view_9, out=buf317)
    del view_9
    buf318 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf319 = reinterpret_tensor(buf316, (8, 197, 3072), (605184, 3072, 1), 0); del buf316  # reuse
    cpp_fused_gelu_gelu_backward_sum_56(c_void_p(buf319.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(addmm_2.data_ptr()), c_void_p(buf318.data_ptr()))
    del addmm_2
    buf320 = buf308; del buf308  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf319, (1576, 3072), (3072, 1), 0), permute_280, out=buf320)
    del permute_280
    buf321 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf319, (3072, 1576), (1, 3072), 0), view_7, out=buf321)
    del view_7
    buf322 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf323 = buf312; del buf312  # reuse
    buf324 = buf311; del buf311  # reuse
    buf325 = empty((768, ), device='cpu', dtype=torch.float32)
    buf326 = empty((768, ), device='cpu', dtype=torch.float32)
    buf327 = buf315; del buf315  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_57(c_void_p(buf327.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(mul_2.data_ptr()), c_void_p(div_23.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()))
    del buf319
    del div_23
    del mul_2
    del primals_11
    buf328 = buf320; del buf320  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf327, (1576, 768), (768, 1), 0), permute_284, out=buf328)
    del permute_284
    buf329 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf327, (768, 1576), (1, 768), 0), view_5, out=buf329)
    del view_5
    buf330 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_58(c_void_p(buf327.data_ptr()), c_void_p(buf330.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf331 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf328, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_2, getitem_3, getitem_4, alias_23, getitem_6, getitem_7, getitem_8, 0, 0, 0.0, False, getitem_11, getitem_12)
    del alias_23
    del buf328
    del getitem_11
    del getitem_12
    del getitem_2
    del getitem_3
    del getitem_4
    del getitem_6
    del getitem_7
    del getitem_8
    buf332 = buf331[0]
    buf333 = buf331[1]
    buf334 = buf331[2]
    del buf331
    buf335 = buf307; del buf307  # reuse
    cpp_fused_clone_59(c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()))
    del buf332
    del buf333
    buf336 = reinterpret_tensor(buf334, (1576, 768), (768, 1), 0); del buf334  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf335, (1576, 2304), (2304, 1), 0), permute_290, out=buf336)
    del permute_290
    buf337 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf335, (2304, 1576), (1, 2304), 0), view_1, out=buf337)
    del view_1
    buf338 = empty((1, 2304), device='cpu', dtype=torch.float32)
    buf339 = buf324; del buf324  # reuse
    buf340 = buf323; del buf323  # reuse
    buf341 = empty((768, ), device='cpu', dtype=torch.float32)
    buf342 = empty((768, ), device='cpu', dtype=torch.float32)
    buf343 = buf327; del buf327  # reuse
    buf344 = empty((1, 197, 768), device='cpu', dtype=torch.float32)
    buf345 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_60(c_void_p(buf343.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(mul.data_ptr()), c_void_p(div_24.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()))
    del buf335
    del buf336
    del buf339
    del buf340
    del div_24
    del mul
    del primals_5
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf346 = aten.convolution_backward(reinterpret_tensor(buf343, (8, 768, 14, 14), (151296, 1, 10752, 768), 768), primals_153, primals_3, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf343
    del primals_153
    del primals_3
    buf347 = buf346[1]
    buf348 = buf346[2]
    return (buf344, buf345, buf347, buf348, buf341, buf342, reinterpret_tensor(buf337, (2304, 768), (768, 1), 0), reinterpret_tensor(buf338, (2304, ), (1, ), 0), reinterpret_tensor(buf329, (768, 768), (768, 1), 0), reinterpret_tensor(buf330, (768, ), (1, ), 0), buf325, buf326, reinterpret_tensor(buf321, (3072, 768), (768, 1), 0), reinterpret_tensor(buf322, (3072, ), (1, ), 0), reinterpret_tensor(buf317, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf318, (768, ), (1, ), 0), buf313, buf314, reinterpret_tensor(buf309, (2304, 768), (768, 1), 0), reinterpret_tensor(buf310, (2304, ), (1, ), 0), reinterpret_tensor(buf301, (768, 768), (768, 1), 0), reinterpret_tensor(buf302, (768, ), (1, ), 0), buf297, buf298, reinterpret_tensor(buf293, (3072, 768), (768, 1), 0), reinterpret_tensor(buf294, (3072, ), (1, ), 0), reinterpret_tensor(buf289, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf290, (768, ), (1, ), 0), buf285, buf286, reinterpret_tensor(buf281, (2304, 768), (768, 1), 0), reinterpret_tensor(buf282, (2304, ), (1, ), 0), reinterpret_tensor(buf273, (768, 768), (768, 1), 0), reinterpret_tensor(buf274, (768, ), (1, ), 0), buf269, buf270, reinterpret_tensor(buf265, (3072, 768), (768, 1), 0), reinterpret_tensor(buf266, (3072, ), (1, ), 0), reinterpret_tensor(buf261, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf262, (768, ), (1, ), 0), buf257, buf258, reinterpret_tensor(buf253, (2304, 768), (768, 1), 0), reinterpret_tensor(buf254, (2304, ), (1, ), 0), reinterpret_tensor(buf245, (768, 768), (768, 1), 0), reinterpret_tensor(buf246, (768, ), (1, ), 0), buf241, buf242, reinterpret_tensor(buf237, (3072, 768), (768, 1), 0), reinterpret_tensor(buf238, (3072, ), (1, ), 0), reinterpret_tensor(buf233, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf234, (768, ), (1, ), 0), buf229, buf230, reinterpret_tensor(buf225, (2304, 768), (768, 1), 0), reinterpret_tensor(buf226, (2304, ), (1, ), 0), reinterpret_tensor(buf217, (768, 768), (768, 1), 0), reinterpret_tensor(buf218, (768, ), (1, ), 0), buf213, buf214, reinterpret_tensor(buf209, (3072, 768), (768, 1), 0), reinterpret_tensor(buf210, (3072, ), (1, ), 0), reinterpret_tensor(buf205, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf206, (768, ), (1, ), 0), buf201, buf202, reinterpret_tensor(buf197, (2304, 768), (768, 1), 0), reinterpret_tensor(buf198, (2304, ), (1, ), 0), reinterpret_tensor(buf189, (768, 768), (768, 1), 0), reinterpret_tensor(buf190, (768, ), (1, ), 0), buf185, buf186, reinterpret_tensor(buf181, (3072, 768), (768, 1), 0), reinterpret_tensor(buf182, (3072, ), (1, ), 0), reinterpret_tensor(buf177, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf178, (768, ), (1, ), 0), buf173, buf174, reinterpret_tensor(buf169, (2304, 768), (768, 1), 0), reinterpret_tensor(buf170, (2304, ), (1, ), 0), reinterpret_tensor(buf161, (768, 768), (768, 1), 0), reinterpret_tensor(buf162, (768, ), (1, ), 0), buf157, buf158, reinterpret_tensor(buf153, (3072, 768), (768, 1), 0), reinterpret_tensor(buf154, (3072, ), (1, ), 0), reinterpret_tensor(buf149, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf150, (768, ), (1, ), 0), buf145, buf146, reinterpret_tensor(buf141, (2304, 768), (768, 1), 0), reinterpret_tensor(buf142, (2304, ), (1, ), 0), reinterpret_tensor(buf133, (768, 768), (768, 1), 0), reinterpret_tensor(buf134, (768, ), (1, ), 0), buf129, buf130, reinterpret_tensor(buf125, (3072, 768), (768, 1), 0), reinterpret_tensor(buf126, (3072, ), (1, ), 0), reinterpret_tensor(buf121, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf122, (768, ), (1, ), 0), buf117, buf118, reinterpret_tensor(buf113, (2304, 768), (768, 1), 0), reinterpret_tensor(buf114, (2304, ), (1, ), 0), reinterpret_tensor(buf105, (768, 768), (768, 1), 0), reinterpret_tensor(buf106, (768, ), (1, ), 0), buf101, buf102, reinterpret_tensor(buf97, (3072, 768), (768, 1), 0), reinterpret_tensor(buf98, (3072, ), (1, ), 0), reinterpret_tensor(buf93, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf94, (768, ), (1, ), 0), buf89, buf90, reinterpret_tensor(buf85, (2304, 768), (768, 1), 0), reinterpret_tensor(buf86, (2304, ), (1, ), 0), reinterpret_tensor(buf77, (768, 768), (768, 1), 0), reinterpret_tensor(buf78, (768, ), (1, ), 0), buf73, buf74, reinterpret_tensor(buf69, (3072, 768), (768, 1), 0), reinterpret_tensor(buf70, (3072, ), (1, ), 0), reinterpret_tensor(buf65, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf66, (768, ), (1, ), 0), buf61, buf62, reinterpret_tensor(buf57, (2304, 768), (768, 1), 0), reinterpret_tensor(buf58, (2304, ), (1, ), 0), reinterpret_tensor(buf49, (768, 768), (768, 1), 0), reinterpret_tensor(buf50, (768, ), (1, ), 0), buf45, buf46, reinterpret_tensor(buf41, (3072, 768), (768, 1), 0), reinterpret_tensor(buf42, (3072, ), (1, ), 0), reinterpret_tensor(buf37, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf38, (768, ), (1, ), 0), buf33, buf34, reinterpret_tensor(buf29, (2304, 768), (768, 1), 0), reinterpret_tensor(buf30, (2304, ), (1, ), 0), reinterpret_tensor(buf21, (768, 768), (768, 1), 0), reinterpret_tensor(buf22, (768, ), (1, ), 0), buf17, buf18, reinterpret_tensor(buf13, (3072, 768), (768, 1), 0), reinterpret_tensor(buf14, (3072, ), (1, ), 0), reinterpret_tensor(buf9, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf10, (768, ), (1, ), 0), buf6, buf7, reinterpret_tensor(buf1, (1000, 768), (768, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((768, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    mul = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_1 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_2 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_4 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_6 = rand_strided((8, 12, 197), (2364, 1, 12), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_8 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_11 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_12 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_5 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_2 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_7 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_2 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_9 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_7 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_11 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_18 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_19 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_20 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_22 = rand_strided((8, 12, 197), (2364, 1, 12), device='cpu', dtype=torch.float32)
    getitem_23 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_24 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_27 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_28 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_15 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_9 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_17 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_6 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_19 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_14 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_21 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_34 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_35 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_36 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_38 = rand_strided((8, 12, 197), (2364, 1, 12), device='cpu', dtype=torch.float32)
    getitem_39 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_40 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_43 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_44 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_25 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_16 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_27 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_29 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_21 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_31 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_50 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_51 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_52 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_54 = rand_strided((8, 12, 197), (2364, 1, 12), device='cpu', dtype=torch.float32)
    getitem_55 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_56 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_59 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_60 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_35 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_23 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_37 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_14 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_39 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_28 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_41 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_66 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_67 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_68 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_70 = rand_strided((8, 12, 197), (2364, 1, 12), device='cpu', dtype=torch.float32)
    getitem_71 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_72 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_75 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_76 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_45 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_30 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_47 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_18 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_49 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_35 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_51 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_82 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_83 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_84 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_86 = rand_strided((8, 12, 197), (2364, 1, 12), device='cpu', dtype=torch.float32)
    getitem_87 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_88 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_91 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_92 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_55 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_37 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_57 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_59 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_42 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_61 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_98 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_99 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_100 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_102 = rand_strided((8, 12, 197), (2364, 1, 12), device='cpu', dtype=torch.float32)
    getitem_103 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_104 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_107 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_108 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_65 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_44 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_67 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_26 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_69 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_49 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_71 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_114 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_115 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_116 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_118 = rand_strided((8, 12, 197), (2364, 1, 12), device='cpu', dtype=torch.float32)
    getitem_119 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_120 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_123 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_124 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_75 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_51 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_77 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_30 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_79 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_56 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_81 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_130 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_131 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_132 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_134 = rand_strided((8, 12, 197), (2364, 1, 12), device='cpu', dtype=torch.float32)
    getitem_135 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_136 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_139 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_140 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_85 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_58 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_87 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_89 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_63 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_91 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_146 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_147 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_148 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_150 = rand_strided((8, 12, 197), (2364, 1, 12), device='cpu', dtype=torch.float32)
    getitem_151 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_152 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_155 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_156 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_95 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_65 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_97 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_38 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_99 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_70 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_101 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_162 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_163 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_164 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_166 = rand_strided((8, 12, 197), (2364, 1, 12), device='cpu', dtype=torch.float32)
    getitem_167 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_168 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_171 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_172 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_105 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_72 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_107 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_42 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_109 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_77 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_111 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_178 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_179 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_180 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cpu', dtype=torch.float32)
    getitem_182 = rand_strided((8, 12, 197), (2364, 1, 12), device='cpu', dtype=torch.float32)
    getitem_183 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_184 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_187 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_188 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_115 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_79 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_117 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_46 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_119 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_84 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    clone_37 = rand_strided((8, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_74 = rand_strided((1000, 768), (768, 1), device='cpu', dtype=torch.float32)
    div = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_78 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_82 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_1 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_86 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_12 = rand_strided((8, 12, 197, 64), (151296, 1, 768, 12), device='cpu', dtype=torch.float32)
    permute_92 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_2 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_96 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_100 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_3 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_104 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_13 = rand_strided((8, 12, 197, 64), (151296, 1, 768, 12), device='cpu', dtype=torch.float32)
    permute_110 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_4 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_114 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_118 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_5 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_122 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_14 = rand_strided((8, 12, 197, 64), (151296, 1, 768, 12), device='cpu', dtype=torch.float32)
    permute_128 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_6 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_132 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_136 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_7 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_140 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_15 = rand_strided((8, 12, 197, 64), (151296, 1, 768, 12), device='cpu', dtype=torch.float32)
    permute_146 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_8 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_150 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_154 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_9 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_158 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_16 = rand_strided((8, 12, 197, 64), (151296, 1, 768, 12), device='cpu', dtype=torch.float32)
    permute_164 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_10 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_168 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_172 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_11 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_176 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_17 = rand_strided((8, 12, 197, 64), (151296, 1, 768, 12), device='cpu', dtype=torch.float32)
    permute_182 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_12 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_186 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_190 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_13 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_194 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_18 = rand_strided((8, 12, 197, 64), (151296, 1, 768, 12), device='cpu', dtype=torch.float32)
    permute_200 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_14 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_204 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_208 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_15 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_212 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_19 = rand_strided((8, 12, 197, 64), (151296, 1, 768, 12), device='cpu', dtype=torch.float32)
    permute_218 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_16 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_222 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_226 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_17 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_230 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_20 = rand_strided((8, 12, 197, 64), (151296, 1, 768, 12), device='cpu', dtype=torch.float32)
    permute_236 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_18 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_240 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_244 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_19 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_248 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_21 = rand_strided((8, 12, 197, 64), (151296, 1, 768, 12), device='cpu', dtype=torch.float32)
    permute_254 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_20 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_258 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_262 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_21 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_266 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_22 = rand_strided((8, 12, 197, 64), (151296, 1, 768, 12), device='cpu', dtype=torch.float32)
    permute_272 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_22 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_276 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_280 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_23 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_284 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    alias_23 = rand_strided((8, 12, 197, 64), (151296, 1, 768, 12), device='cpu', dtype=torch.float32)
    permute_290 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_3, primals_5, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_47, primals_53, primals_59, primals_65, primals_71, primals_77, primals_83, primals_89, primals_95, primals_101, primals_107, primals_113, primals_119, primals_125, primals_131, primals_137, primals_143, primals_149, primals_153, mul, view_1, getitem_2, getitem_3, getitem_4, getitem_6, getitem_7, getitem_8, getitem_11, getitem_12, view_5, mul_2, view_7, addmm_2, view_9, mul_7, view_11, getitem_18, getitem_19, getitem_20, getitem_22, getitem_23, getitem_24, getitem_27, getitem_28, view_15, mul_9, view_17, addmm_6, view_19, mul_14, view_21, getitem_34, getitem_35, getitem_36, getitem_38, getitem_39, getitem_40, getitem_43, getitem_44, view_25, mul_16, view_27, addmm_10, view_29, mul_21, view_31, getitem_50, getitem_51, getitem_52, getitem_54, getitem_55, getitem_56, getitem_59, getitem_60, view_35, mul_23, view_37, addmm_14, view_39, mul_28, view_41, getitem_66, getitem_67, getitem_68, getitem_70, getitem_71, getitem_72, getitem_75, getitem_76, view_45, mul_30, view_47, addmm_18, view_49, mul_35, view_51, getitem_82, getitem_83, getitem_84, getitem_86, getitem_87, getitem_88, getitem_91, getitem_92, view_55, mul_37, view_57, addmm_22, view_59, mul_42, view_61, getitem_98, getitem_99, getitem_100, getitem_102, getitem_103, getitem_104, getitem_107, getitem_108, view_65, mul_44, view_67, addmm_26, view_69, mul_49, view_71, getitem_114, getitem_115, getitem_116, getitem_118, getitem_119, getitem_120, getitem_123, getitem_124, view_75, mul_51, view_77, addmm_30, view_79, mul_56, view_81, getitem_130, getitem_131, getitem_132, getitem_134, getitem_135, getitem_136, getitem_139, getitem_140, view_85, mul_58, view_87, addmm_34, view_89, mul_63, view_91, getitem_146, getitem_147, getitem_148, getitem_150, getitem_151, getitem_152, getitem_155, getitem_156, view_95, mul_65, view_97, addmm_38, view_99, mul_70, view_101, getitem_162, getitem_163, getitem_164, getitem_166, getitem_167, getitem_168, getitem_171, getitem_172, view_105, mul_72, view_107, addmm_42, view_109, mul_77, view_111, getitem_178, getitem_179, getitem_180, getitem_182, getitem_183, getitem_184, getitem_187, getitem_188, view_115, mul_79, view_117, addmm_46, view_119, mul_84, clone_37, permute_74, div, permute_78, permute_82, div_1, permute_86, alias_12, permute_92, div_2, permute_96, permute_100, div_3, permute_104, alias_13, permute_110, div_4, permute_114, permute_118, div_5, permute_122, alias_14, permute_128, div_6, permute_132, permute_136, div_7, permute_140, alias_15, permute_146, div_8, permute_150, permute_154, div_9, permute_158, alias_16, permute_164, div_10, permute_168, permute_172, div_11, permute_176, alias_17, permute_182, div_12, permute_186, permute_190, div_13, permute_194, alias_18, permute_200, div_14, permute_204, permute_208, div_15, permute_212, alias_19, permute_218, div_16, permute_222, permute_226, div_17, permute_230, alias_20, permute_236, div_18, permute_240, permute_244, div_19, permute_248, alias_21, permute_254, div_20, permute_258, permute_262, div_21, permute_266, alias_22, permute_272, div_22, permute_276, permute_280, div_23, permute_284, alias_23, permute_290, div_24, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('vit_base_patch16_224', benchmark_compiled_module)
