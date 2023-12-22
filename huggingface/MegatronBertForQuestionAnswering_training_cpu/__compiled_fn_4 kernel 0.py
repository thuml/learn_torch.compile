
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


cpp_fused_nll_loss_backward_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_nll_loss_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__log_softmax_backward_data_cat_div_nll_loss_backward_nll_loss_forward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       const float* in_ptr4,
                       const bool* in_ptr5,
                       const bool* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        {
            float tmp_acc0 = 0;
            float tmp_acc1 = 0;
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(0L)];
                auto tmp2 = in_ptr2[static_cast<long>(0L)];
                auto tmp5 = in_ptr3[static_cast<long>(0L)];
                auto tmp12 = in_ptr4[static_cast<long>(x0)];
                auto tmp13 = in_ptr5[static_cast<long>(0L)];
                auto tmp14 = in_ptr6[static_cast<long>(0L)];
                auto tmp3 = static_cast<float>(2.0);
                auto tmp4 = tmp2 / tmp3;
                auto tmp6 = c10::convert<long>(tmp5);
                auto tmp7 = c10::convert<float>(tmp6);
                auto tmp8 = tmp4 / tmp7;
                auto tmp9 = static_cast<float>(0.0);
                auto tmp10 = tmp1 ? tmp8 : tmp9;
                auto tmp11 = decltype(tmp0)(tmp0 * tmp10);
                auto tmp15 = c10::convert<long>(tmp14);
                auto tmp16 = c10::convert<float>(tmp15);
                auto tmp17 = tmp4 / tmp16;
                auto tmp18 = tmp13 ? tmp17 : tmp9;
                auto tmp19 = decltype(tmp12)(tmp12 * tmp18);
                tmp_acc0 = tmp_acc0 + tmp11;
                tmp_acc1 = tmp_acc1 + tmp19;
            }
            out_ptr0[static_cast<long>(0L)] = tmp_acc0;
            out_ptr1[static_cast<long>(0L)] = tmp_acc1;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(1);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr7[static_cast<long>(x0)];
                    auto tmp7 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = in_ptr5[static_cast<long>(0L)];
                    auto tmp9 = in_ptr2[static_cast<long>(0L)];
                    auto tmp10 = static_cast<float>(2.0);
                    auto tmp11 = tmp9 / tmp10;
                    auto tmp12 = in_ptr6[static_cast<long>(0L)];
                    auto tmp13 = c10::convert<long>(tmp12);
                    auto tmp14 = c10::convert<float>(tmp13);
                    auto tmp15 = tmp11 / tmp14;
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = tmp8 ? tmp15 : tmp16;
                    auto tmp18 = decltype(tmp7)(tmp7 * tmp17);
                    auto tmp19 = in_ptr8[static_cast<long>(x0)];
                    auto tmp20 = std::exp(tmp19);
                    auto tmp21 = out_ptr1[static_cast<long>(0L)];
                    auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                    auto tmp23 = decltype(tmp18)(tmp18 - tmp22);
                    auto tmp24 = decltype(tmp6)(tmp6 + tmp23);
                    return tmp24;
                }
                ;
                auto tmp25 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp26 = tmp0 >= tmp3;
                auto tmp27 = static_cast<long>(2);
                auto tmp28 = tmp0 < tmp27;
                auto tmp29 = [&]
                {
                    auto tmp30 = in_ptr9[static_cast<long>(x0)];
                    auto tmp31 = in_ptr0[static_cast<long>(x0)];
                    auto tmp32 = in_ptr1[static_cast<long>(0L)];
                    auto tmp33 = in_ptr2[static_cast<long>(0L)];
                    auto tmp34 = static_cast<float>(2.0);
                    auto tmp35 = tmp33 / tmp34;
                    auto tmp36 = in_ptr3[static_cast<long>(0L)];
                    auto tmp37 = c10::convert<long>(tmp36);
                    auto tmp38 = c10::convert<float>(tmp37);
                    auto tmp39 = tmp35 / tmp38;
                    auto tmp40 = static_cast<float>(0.0);
                    auto tmp41 = tmp32 ? tmp39 : tmp40;
                    auto tmp42 = decltype(tmp31)(tmp31 * tmp41);
                    auto tmp43 = in_ptr10[static_cast<long>(x0)];
                    auto tmp44 = std::exp(tmp43);
                    auto tmp45 = out_ptr0[static_cast<long>(0L)];
                    auto tmp46 = decltype(tmp44)(tmp44 * tmp45);
                    auto tmp47 = decltype(tmp42)(tmp42 - tmp46);
                    auto tmp48 = decltype(tmp30)(tmp30 + tmp47);
                    return tmp48;
                }
                ;
                auto tmp49 = tmp26 ? tmp29() : static_cast<decltype(tmp29())>(0.0);
                auto tmp50 = tmp4 ? tmp25 : tmp49;
                out_ptr2[static_cast<long>(x1 + (2L*x0))] = tmp50;
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
    {
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (2L*x1))];
                    tmp_acc0 = tmp_acc0 + tmp0;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
            }
        }
    }
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_gelu_gelu_backward_sum_4 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_6 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_14 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_20 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_22 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
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
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_28 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_30 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_38 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_44 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
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
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_52 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_54 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_60 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_62 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_68 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_76 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_77 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_78 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_82 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_84 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_85 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_86 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_90 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_92 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_93 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_94 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_98 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
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
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_102 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_106 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_109 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_110 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_114 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_115 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_116 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_117 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_118 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_122 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_124 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_125 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_126 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_127 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_129 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_130 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_131 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_132 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_133 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_134 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_135 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_136 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_137 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_138 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_139 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_140 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_141 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_142 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_143 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_144 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_145 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_146 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_147 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_148 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_149 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_150 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_151 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_152 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_153 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_154 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_155 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_157 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_158 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_159 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_160 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_161 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_162 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_163 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_164 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_165 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_166 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_167 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_168 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_169 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_170 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_171 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_172 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_173 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_174 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_175 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_176 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_177 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_178 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_179 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_180 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_181 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_182 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_183 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_184 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_185 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_186 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_187 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp16 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp19 = tmp14 - tmp18;
                    auto tmp20 = at::vec::Vectorized<float>(tmp1);
                    auto tmp21 = tmp20 * tmp19;
                    auto tmp22 = tmp0 + tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr7[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_188 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_189 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(1024.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_190 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused_191 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (512L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (512L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (512L*x0))];
                    auto tmp6 = in_ptr2[static_cast<long>(x1 + (512L*x0))];
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                    auto tmp10 = decltype(tmp7)(tmp7 - tmp9);
                    in_out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp10;
                }
            }
        }
    }
}
''')


cpp_fused_view_192 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_193 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.3535533905932738);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_194 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_sum_195 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const long* in_ptr7,
                       const bool* in_ptr8,
                       const long* in_ptr9,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (1024L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = in_ptr1[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp3 = in_ptr2[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp5 = in_ptr3[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp7 = in_ptr4[static_cast<long>(x1)];
                    auto tmp11 = out_ptr1[static_cast<long>(x0)];
                    auto tmp13 = in_ptr5[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp14 = out_ptr2[static_cast<long>(x0)];
                    auto tmp19 = in_ptr7[static_cast<long>(x0)];
                    auto tmp22 = in_ptr8[static_cast<long>(x1 + (1024L*x0))];
                    auto tmp29 = in_ptr9[static_cast<long>(x0)];
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp12 = decltype(tmp10)(tmp10 - tmp11);
                    auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                    auto tmp16 = decltype(tmp12)(tmp12 - tmp15);
                    auto tmp17 = decltype(tmp1)(tmp1 * tmp16);
                    auto tmp18 = decltype(tmp0)(tmp0 + tmp17);
                    auto tmp20 = static_cast<long>(-1);
                    auto tmp21 = tmp19 == tmp20;
                    auto tmp23 = c10::convert<float>(tmp22);
                    auto tmp24 = static_cast<float>(1.1111111111111112);
                    auto tmp25 = decltype(tmp23)(tmp23 * tmp24);
                    auto tmp26 = decltype(tmp18)(tmp18 * tmp25);
                    auto tmp27 = static_cast<float>(0.0);
                    auto tmp28 = tmp21 ? tmp27 : tmp26;
                    auto tmp30 = static_cast<long>(0);
                    auto tmp31 = tmp29 == tmp30;
                    auto tmp32 = tmp31 ? tmp27 : tmp26;
                    in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp18;
                    out_ptr5[static_cast<long>(x1 + (1024L*x0))] = tmp28;
                    out_ptr6[static_cast<long>(x1 + (1024L*x0))] = tmp32;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr7 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_native_dropout_backward_nll_loss_forward_196 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = static_cast<float>(1.1111111111111112);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                auto tmp6 = static_cast<bool>(0);
                auto tmp7 = static_cast<float>(0.0);
                auto tmp8 = tmp6 ? tmp7 : tmp5;
                in_out_ptr0[static_cast<long>(x0)] = tmp8;
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_197 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(29753344L); x0+=static_cast<long>(8L))
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
    primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_206, primals_212, primals_222, primals_228, primals_238, primals_244, primals_254, primals_260, primals_270, primals_276, primals_286, primals_292, primals_302, primals_308, primals_318, primals_324, primals_334, primals_340, primals_350, primals_356, primals_366, primals_372, primals_382, primals_388, primals_393, full_default, slice_3, getitem_1, mul_1, view, getitem_293, permute_default_139, permute_default_140, alias_default_47, permute_default_141, permute_default_142, view_16, getitem_7, mul_3, view_18, addmm_4, view_20, getitem_11, mul_8, view_22, getitem_291, permute_default_133, permute_default_134, alias_default_45, permute_default_135, permute_default_136, view_38, getitem_17, mul_10, view_40, addmm_10, view_42, getitem_21, mul_15, view_44, getitem_289, permute_default_127, permute_default_128, alias_default_43, permute_default_129, permute_default_130, view_60, getitem_27, mul_17, view_62, addmm_16, view_64, getitem_31, mul_22, view_66, getitem_287, permute_default_121, permute_default_122, alias_default_41, permute_default_123, permute_default_124, view_82, getitem_37, mul_24, view_84, addmm_22, view_86, getitem_41, mul_29, view_88, getitem_285, permute_default_115, permute_default_116, alias_default_39, permute_default_117, permute_default_118, view_104, getitem_47, mul_31, view_106, addmm_28, view_108, getitem_51, mul_36, view_110, getitem_283, permute_default_109, permute_default_110, alias_default_37, permute_default_111, permute_default_112, view_126, getitem_57, mul_38, view_128, addmm_34, view_130, getitem_61, mul_43, view_132, getitem_281, permute_default_103, permute_default_104, alias_default_35, permute_default_105, permute_default_106, view_148, getitem_67, mul_45, view_150, addmm_40, view_152, getitem_71, mul_50, view_154, getitem_279, permute_default_97, permute_default_98, alias_default_33, permute_default_99, permute_default_100, view_170, getitem_77, mul_52, view_172, addmm_46, view_174, getitem_81, mul_57, view_176, getitem_277, permute_default_91, permute_default_92, alias_default_31, permute_default_93, permute_default_94, view_192, getitem_87, mul_59, view_194, addmm_52, view_196, getitem_91, mul_64, view_198, getitem_275, permute_default_85, permute_default_86, alias_default_29, permute_default_87, permute_default_88, view_214, getitem_97, mul_66, view_216, addmm_58, view_218, getitem_101, mul_71, view_220, getitem_273, permute_default_79, permute_default_80, alias_default_27, permute_default_81, permute_default_82, view_236, getitem_107, mul_73, view_238, addmm_64, view_240, getitem_111, mul_78, view_242, getitem_271, permute_default_73, permute_default_74, alias_default_25, permute_default_75, permute_default_76, view_258, getitem_117, mul_80, view_260, addmm_70, view_262, getitem_121, mul_85, view_264, getitem_269, permute_default_67, permute_default_68, alias_default_23, permute_default_69, permute_default_70, view_280, getitem_127, mul_87, view_282, addmm_76, view_284, getitem_131, mul_92, view_286, getitem_267, permute_default_61, permute_default_62, alias_default_21, permute_default_63, permute_default_64, view_302, getitem_137, mul_94, view_304, addmm_82, view_306, getitem_141, mul_99, view_308, getitem_265, permute_default_55, permute_default_56, alias_default_19, permute_default_57, permute_default_58, view_324, getitem_147, mul_101, view_326, addmm_88, view_328, getitem_151, mul_106, view_330, getitem_263, permute_default_49, permute_default_50, alias_default_17, permute_default_51, permute_default_52, view_346, getitem_157, mul_108, view_348, addmm_94, view_350, getitem_161, mul_113, view_352, getitem_261, permute_default_43, permute_default_44, alias_default_15, permute_default_45, permute_default_46, view_368, getitem_167, mul_115, view_370, addmm_100, view_372, getitem_171, mul_120, view_374, getitem_259, permute_default_37, permute_default_38, alias_default_13, permute_default_39, permute_default_40, view_390, getitem_177, mul_122, view_392, addmm_106, view_394, getitem_181, mul_127, view_396, getitem_257, permute_default_31, permute_default_32, alias_default_11, permute_default_33, permute_default_34, view_412, getitem_187, mul_129, view_414, addmm_112, view_416, getitem_191, mul_134, view_418, getitem_255, permute_default_25, permute_default_26, alias_default_9, permute_default_27, permute_default_28, view_434, getitem_197, mul_136, view_436, addmm_118, view_438, getitem_201, mul_141, view_440, getitem_253, permute_default_19, permute_default_20, alias_default_7, permute_default_21, permute_default_22, view_456, getitem_207, mul_143, view_458, addmm_124, view_460, getitem_211, mul_148, view_462, getitem_251, permute_default_13, permute_default_14, alias_default_5, permute_default_15, permute_default_16, view_478, getitem_217, mul_150, view_480, addmm_130, view_482, getitem_221, mul_155, view_484, getitem_249, permute_default_7, permute_default_8, alias_default_3, permute_default_9, permute_default_10, view_500, getitem_227, mul_157, view_502, addmm_136, view_504, getitem_231, mul_162, view_506, getitem_247, permute_default_1, permute_default_2, alias_default_1, permute_default_3, permute_default_4, view_522, getitem_237, mul_164, view_524, addmm_142, view_526, getitem_241, mul_169, view_528, sub_75, ne, sub_77, ne_3, ne_6, where_4, ne_8, where_6, permute_265, div_54, permute_269, permute_273, div_55, permute_277, permute_289, permute_294, permute_298, div_57, permute_302, permute_306, div_58, permute_310, permute_322, permute_327, permute_331, div_60, permute_335, permute_339, div_61, permute_343, permute_355, permute_360, permute_364, div_63, permute_368, permute_372, div_64, permute_376, permute_388, permute_393, permute_397, div_66, permute_401, permute_405, div_67, permute_409, permute_421, permute_426, permute_430, div_69, permute_434, permute_438, div_70, permute_442, permute_454, permute_459, permute_463, div_72, permute_467, permute_471, div_73, permute_475, permute_487, permute_492, permute_496, div_75, permute_500, permute_504, div_76, permute_508, permute_520, permute_525, permute_529, div_78, permute_533, permute_537, div_79, permute_541, permute_553, permute_558, permute_562, div_81, permute_566, permute_570, div_82, permute_574, permute_586, permute_591, permute_595, div_84, permute_599, permute_603, div_85, permute_607, permute_619, permute_624, permute_628, div_87, permute_632, permute_636, div_88, permute_640, permute_652, permute_657, permute_661, div_90, permute_665, permute_669, div_91, permute_673, permute_685, permute_690, permute_694, div_93, permute_698, permute_702, div_94, permute_706, permute_718, permute_723, permute_727, div_96, permute_731, permute_735, div_97, permute_739, permute_751, permute_756, permute_760, div_99, permute_764, permute_768, div_100, permute_772, permute_784, permute_789, permute_793, div_102, permute_797, permute_801, div_103, permute_805, permute_817, permute_822, permute_826, div_105, permute_830, permute_834, div_106, permute_838, permute_850, permute_855, permute_859, div_108, permute_863, permute_867, div_109, permute_871, permute_883, permute_888, permute_892, div_111, permute_896, permute_900, div_112, permute_904, permute_916, permute_921, permute_925, div_114, permute_929, permute_933, div_115, permute_937, permute_949, permute_954, permute_958, div_117, permute_962, permute_966, div_118, permute_970, permute_982, permute_987, permute_991, div_120, permute_995, permute_999, div_121, permute_1003, permute_1015, permute_1020, permute_1024, div_123, permute_1028, permute_1032, div_124, permute_1036, permute_1048, permute_1053, permute_1057, div_126, tangents_1, tangents_2, tangents_3 = args
    args.clear()
    assert_size_stride(primals_4, (1024, ), (1, ))
    assert_size_stride(primals_14, (1024, ), (1, ))
    assert_size_stride(primals_20, (1024, ), (1, ))
    assert_size_stride(primals_30, (1024, ), (1, ))
    assert_size_stride(primals_36, (1024, ), (1, ))
    assert_size_stride(primals_46, (1024, ), (1, ))
    assert_size_stride(primals_52, (1024, ), (1, ))
    assert_size_stride(primals_62, (1024, ), (1, ))
    assert_size_stride(primals_68, (1024, ), (1, ))
    assert_size_stride(primals_78, (1024, ), (1, ))
    assert_size_stride(primals_84, (1024, ), (1, ))
    assert_size_stride(primals_94, (1024, ), (1, ))
    assert_size_stride(primals_100, (1024, ), (1, ))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_116, (1024, ), (1, ))
    assert_size_stride(primals_126, (1024, ), (1, ))
    assert_size_stride(primals_132, (1024, ), (1, ))
    assert_size_stride(primals_142, (1024, ), (1, ))
    assert_size_stride(primals_148, (1024, ), (1, ))
    assert_size_stride(primals_158, (1024, ), (1, ))
    assert_size_stride(primals_164, (1024, ), (1, ))
    assert_size_stride(primals_174, (1024, ), (1, ))
    assert_size_stride(primals_180, (1024, ), (1, ))
    assert_size_stride(primals_190, (1024, ), (1, ))
    assert_size_stride(primals_196, (1024, ), (1, ))
    assert_size_stride(primals_206, (1024, ), (1, ))
    assert_size_stride(primals_212, (1024, ), (1, ))
    assert_size_stride(primals_222, (1024, ), (1, ))
    assert_size_stride(primals_228, (1024, ), (1, ))
    assert_size_stride(primals_238, (1024, ), (1, ))
    assert_size_stride(primals_244, (1024, ), (1, ))
    assert_size_stride(primals_254, (1024, ), (1, ))
    assert_size_stride(primals_260, (1024, ), (1, ))
    assert_size_stride(primals_270, (1024, ), (1, ))
    assert_size_stride(primals_276, (1024, ), (1, ))
    assert_size_stride(primals_286, (1024, ), (1, ))
    assert_size_stride(primals_292, (1024, ), (1, ))
    assert_size_stride(primals_302, (1024, ), (1, ))
    assert_size_stride(primals_308, (1024, ), (1, ))
    assert_size_stride(primals_318, (1024, ), (1, ))
    assert_size_stride(primals_324, (1024, ), (1, ))
    assert_size_stride(primals_334, (1024, ), (1, ))
    assert_size_stride(primals_340, (1024, ), (1, ))
    assert_size_stride(primals_350, (1024, ), (1, ))
    assert_size_stride(primals_356, (1024, ), (1, ))
    assert_size_stride(primals_366, (1024, ), (1, ))
    assert_size_stride(primals_372, (1024, ), (1, ))
    assert_size_stride(primals_382, (1024, ), (1, ))
    assert_size_stride(primals_388, (1024, ), (1, ))
    assert_size_stride(primals_393, (1, 512), (512, 1))
    assert_size_stride(full_default, (1, 512), (512, 1))
    assert_size_stride(slice_3, (1, 512), (512, 1))
    assert_size_stride(getitem_1, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_1, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view, (512, 1024), (1024, 1))
    assert_size_stride(getitem_293, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_139, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_140, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_47, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_141, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_142, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_16, (512, 1024), (1024, 1))
    assert_size_stride(getitem_7, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_3, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_18, (512, 1024), (1024, 1))
    assert_size_stride(addmm_4, (512, 4096), (4096, 1))
    assert_size_stride(view_20, (512, 4096), (4096, 1))
    assert_size_stride(getitem_11, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_8, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_22, (512, 1024), (1024, 1))
    assert_size_stride(getitem_291, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_133, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_134, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_45, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_135, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_136, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_38, (512, 1024), (1024, 1))
    assert_size_stride(getitem_17, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_10, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_40, (512, 1024), (1024, 1))
    assert_size_stride(addmm_10, (512, 4096), (4096, 1))
    assert_size_stride(view_42, (512, 4096), (4096, 1))
    assert_size_stride(getitem_21, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_15, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_44, (512, 1024), (1024, 1))
    assert_size_stride(getitem_289, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_127, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_128, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_43, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_129, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_130, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_60, (512, 1024), (1024, 1))
    assert_size_stride(getitem_27, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_17, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_62, (512, 1024), (1024, 1))
    assert_size_stride(addmm_16, (512, 4096), (4096, 1))
    assert_size_stride(view_64, (512, 4096), (4096, 1))
    assert_size_stride(getitem_31, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_22, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_66, (512, 1024), (1024, 1))
    assert_size_stride(getitem_287, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_121, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_122, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_41, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_123, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_124, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_82, (512, 1024), (1024, 1))
    assert_size_stride(getitem_37, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_24, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_84, (512, 1024), (1024, 1))
    assert_size_stride(addmm_22, (512, 4096), (4096, 1))
    assert_size_stride(view_86, (512, 4096), (4096, 1))
    assert_size_stride(getitem_41, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_29, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_88, (512, 1024), (1024, 1))
    assert_size_stride(getitem_285, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_115, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_116, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_39, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_117, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_118, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_104, (512, 1024), (1024, 1))
    assert_size_stride(getitem_47, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_31, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_106, (512, 1024), (1024, 1))
    assert_size_stride(addmm_28, (512, 4096), (4096, 1))
    assert_size_stride(view_108, (512, 4096), (4096, 1))
    assert_size_stride(getitem_51, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_36, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_110, (512, 1024), (1024, 1))
    assert_size_stride(getitem_283, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_109, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_110, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_37, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_111, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_112, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_126, (512, 1024), (1024, 1))
    assert_size_stride(getitem_57, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_38, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_128, (512, 1024), (1024, 1))
    assert_size_stride(addmm_34, (512, 4096), (4096, 1))
    assert_size_stride(view_130, (512, 4096), (4096, 1))
    assert_size_stride(getitem_61, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_43, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_132, (512, 1024), (1024, 1))
    assert_size_stride(getitem_281, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_103, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_104, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_35, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_105, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_106, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_148, (512, 1024), (1024, 1))
    assert_size_stride(getitem_67, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_45, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_150, (512, 1024), (1024, 1))
    assert_size_stride(addmm_40, (512, 4096), (4096, 1))
    assert_size_stride(view_152, (512, 4096), (4096, 1))
    assert_size_stride(getitem_71, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_50, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_154, (512, 1024), (1024, 1))
    assert_size_stride(getitem_279, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_97, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_98, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_33, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_99, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_100, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_170, (512, 1024), (1024, 1))
    assert_size_stride(getitem_77, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_52, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_172, (512, 1024), (1024, 1))
    assert_size_stride(addmm_46, (512, 4096), (4096, 1))
    assert_size_stride(view_174, (512, 4096), (4096, 1))
    assert_size_stride(getitem_81, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_57, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_176, (512, 1024), (1024, 1))
    assert_size_stride(getitem_277, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_91, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_92, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_31, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_93, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_94, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_192, (512, 1024), (1024, 1))
    assert_size_stride(getitem_87, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_59, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_194, (512, 1024), (1024, 1))
    assert_size_stride(addmm_52, (512, 4096), (4096, 1))
    assert_size_stride(view_196, (512, 4096), (4096, 1))
    assert_size_stride(getitem_91, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_64, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_198, (512, 1024), (1024, 1))
    assert_size_stride(getitem_275, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_85, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_86, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_29, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_87, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_88, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_214, (512, 1024), (1024, 1))
    assert_size_stride(getitem_97, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_66, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_216, (512, 1024), (1024, 1))
    assert_size_stride(addmm_58, (512, 4096), (4096, 1))
    assert_size_stride(view_218, (512, 4096), (4096, 1))
    assert_size_stride(getitem_101, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_71, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_220, (512, 1024), (1024, 1))
    assert_size_stride(getitem_273, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_79, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_80, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_27, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_81, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_82, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_236, (512, 1024), (1024, 1))
    assert_size_stride(getitem_107, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_73, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_238, (512, 1024), (1024, 1))
    assert_size_stride(addmm_64, (512, 4096), (4096, 1))
    assert_size_stride(view_240, (512, 4096), (4096, 1))
    assert_size_stride(getitem_111, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_78, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_242, (512, 1024), (1024, 1))
    assert_size_stride(getitem_271, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_73, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_74, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_25, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_75, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_76, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_258, (512, 1024), (1024, 1))
    assert_size_stride(getitem_117, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_80, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_260, (512, 1024), (1024, 1))
    assert_size_stride(addmm_70, (512, 4096), (4096, 1))
    assert_size_stride(view_262, (512, 4096), (4096, 1))
    assert_size_stride(getitem_121, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_85, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_264, (512, 1024), (1024, 1))
    assert_size_stride(getitem_269, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_67, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_68, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_23, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_69, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_70, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_280, (512, 1024), (1024, 1))
    assert_size_stride(getitem_127, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_87, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_282, (512, 1024), (1024, 1))
    assert_size_stride(addmm_76, (512, 4096), (4096, 1))
    assert_size_stride(view_284, (512, 4096), (4096, 1))
    assert_size_stride(getitem_131, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_92, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_286, (512, 1024), (1024, 1))
    assert_size_stride(getitem_267, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_61, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_62, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_21, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_63, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_64, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_302, (512, 1024), (1024, 1))
    assert_size_stride(getitem_137, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_94, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_304, (512, 1024), (1024, 1))
    assert_size_stride(addmm_82, (512, 4096), (4096, 1))
    assert_size_stride(view_306, (512, 4096), (4096, 1))
    assert_size_stride(getitem_141, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_99, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_308, (512, 1024), (1024, 1))
    assert_size_stride(getitem_265, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_55, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_56, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_19, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_57, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_58, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_324, (512, 1024), (1024, 1))
    assert_size_stride(getitem_147, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_101, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_326, (512, 1024), (1024, 1))
    assert_size_stride(addmm_88, (512, 4096), (4096, 1))
    assert_size_stride(view_328, (512, 4096), (4096, 1))
    assert_size_stride(getitem_151, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_106, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_330, (512, 1024), (1024, 1))
    assert_size_stride(getitem_263, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_49, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_50, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_17, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_51, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_52, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_346, (512, 1024), (1024, 1))
    assert_size_stride(getitem_157, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_108, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_348, (512, 1024), (1024, 1))
    assert_size_stride(addmm_94, (512, 4096), (4096, 1))
    assert_size_stride(view_350, (512, 4096), (4096, 1))
    assert_size_stride(getitem_161, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_113, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_352, (512, 1024), (1024, 1))
    assert_size_stride(getitem_261, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_43, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_44, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_15, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_45, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_46, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_368, (512, 1024), (1024, 1))
    assert_size_stride(getitem_167, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_115, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_370, (512, 1024), (1024, 1))
    assert_size_stride(addmm_100, (512, 4096), (4096, 1))
    assert_size_stride(view_372, (512, 4096), (4096, 1))
    assert_size_stride(getitem_171, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_120, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_374, (512, 1024), (1024, 1))
    assert_size_stride(getitem_259, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_37, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_38, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_13, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_39, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_40, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_390, (512, 1024), (1024, 1))
    assert_size_stride(getitem_177, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_122, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_392, (512, 1024), (1024, 1))
    assert_size_stride(addmm_106, (512, 4096), (4096, 1))
    assert_size_stride(view_394, (512, 4096), (4096, 1))
    assert_size_stride(getitem_181, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_127, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_396, (512, 1024), (1024, 1))
    assert_size_stride(getitem_257, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_31, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_32, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_11, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_33, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_34, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_412, (512, 1024), (1024, 1))
    assert_size_stride(getitem_187, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_129, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_414, (512, 1024), (1024, 1))
    assert_size_stride(addmm_112, (512, 4096), (4096, 1))
    assert_size_stride(view_416, (512, 4096), (4096, 1))
    assert_size_stride(getitem_191, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_134, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_418, (512, 1024), (1024, 1))
    assert_size_stride(getitem_255, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_25, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_26, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_9, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_27, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_28, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_434, (512, 1024), (1024, 1))
    assert_size_stride(getitem_197, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_136, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_436, (512, 1024), (1024, 1))
    assert_size_stride(addmm_118, (512, 4096), (4096, 1))
    assert_size_stride(view_438, (512, 4096), (4096, 1))
    assert_size_stride(getitem_201, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_141, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_440, (512, 1024), (1024, 1))
    assert_size_stride(getitem_253, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_19, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_20, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_7, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_21, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_22, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_456, (512, 1024), (1024, 1))
    assert_size_stride(getitem_207, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_143, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_458, (512, 1024), (1024, 1))
    assert_size_stride(addmm_124, (512, 4096), (4096, 1))
    assert_size_stride(view_460, (512, 4096), (4096, 1))
    assert_size_stride(getitem_211, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_148, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_462, (512, 1024), (1024, 1))
    assert_size_stride(getitem_251, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_13, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_14, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_5, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_15, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_16, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_478, (512, 1024), (1024, 1))
    assert_size_stride(getitem_217, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_150, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_480, (512, 1024), (1024, 1))
    assert_size_stride(addmm_130, (512, 4096), (4096, 1))
    assert_size_stride(view_482, (512, 4096), (4096, 1))
    assert_size_stride(getitem_221, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_155, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_484, (512, 1024), (1024, 1))
    assert_size_stride(getitem_249, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_7, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_8, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_3, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_9, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_10, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_500, (512, 1024), (1024, 1))
    assert_size_stride(getitem_227, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_157, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_502, (512, 1024), (1024, 1))
    assert_size_stride(addmm_136, (512, 4096), (4096, 1))
    assert_size_stride(view_504, (512, 4096), (4096, 1))
    assert_size_stride(getitem_231, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_162, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_506, (512, 1024), (1024, 1))
    assert_size_stride(getitem_247, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_1, (16, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_2, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_1, (1, 16, 512, 512), (4194304, 262144, 512, 1))
    assert_size_stride(permute_default_3, (16, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_4, (16, 512, 64), (32768, 64, 1))
    assert_size_stride(view_522, (512, 1024), (1024, 1))
    assert_size_stride(getitem_237, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_164, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_524, (512, 1024), (1024, 1))
    assert_size_stride(addmm_142, (512, 4096), (4096, 1))
    assert_size_stride(view_526, (512, 4096), (4096, 1))
    assert_size_stride(getitem_241, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(mul_169, (1, 512, 1024), (524288, 1024, 1))
    assert_size_stride(view_528, (512, 1024), (1024, 1))
    assert_size_stride(sub_75, (1, 512), (512, 1))
    assert_size_stride(ne, (1, ), (1, ))
    assert_size_stride(sub_77, (1, 512), (512, 1))
    assert_size_stride(ne_3, (1, ), (1, ))
    assert_size_stride(ne_6, (1, 1), (1, 1))
    assert_size_stride(where_4, (1, 1), (1, 1))
    assert_size_stride(ne_8, (1, 1), (1, 1))
    assert_size_stride(where_6, (1, 1), (1, 1))
    assert_size_stride(permute_265, (2, 1024), (1024, 1))
    assert_size_stride(div_54, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_269, (1024, 4096), (4096, 1))
    assert_size_stride(permute_273, (4096, 1024), (1024, 1))
    assert_size_stride(div_55, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_277, (1024, 1024), (1024, 1))
    assert_size_stride(permute_289, (1024, 1024), (1024, 1))
    assert_size_stride(permute_294, (1024, 1024), (1024, 1))
    assert_size_stride(permute_298, (1024, 1024), (1024, 1))
    assert_size_stride(div_57, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_302, (1024, 4096), (4096, 1))
    assert_size_stride(permute_306, (4096, 1024), (1024, 1))
    assert_size_stride(div_58, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_310, (1024, 1024), (1024, 1))
    assert_size_stride(permute_322, (1024, 1024), (1024, 1))
    assert_size_stride(permute_327, (1024, 1024), (1024, 1))
    assert_size_stride(permute_331, (1024, 1024), (1024, 1))
    assert_size_stride(div_60, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_335, (1024, 4096), (4096, 1))
    assert_size_stride(permute_339, (4096, 1024), (1024, 1))
    assert_size_stride(div_61, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_343, (1024, 1024), (1024, 1))
    assert_size_stride(permute_355, (1024, 1024), (1024, 1))
    assert_size_stride(permute_360, (1024, 1024), (1024, 1))
    assert_size_stride(permute_364, (1024, 1024), (1024, 1))
    assert_size_stride(div_63, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_368, (1024, 4096), (4096, 1))
    assert_size_stride(permute_372, (4096, 1024), (1024, 1))
    assert_size_stride(div_64, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_376, (1024, 1024), (1024, 1))
    assert_size_stride(permute_388, (1024, 1024), (1024, 1))
    assert_size_stride(permute_393, (1024, 1024), (1024, 1))
    assert_size_stride(permute_397, (1024, 1024), (1024, 1))
    assert_size_stride(div_66, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_401, (1024, 4096), (4096, 1))
    assert_size_stride(permute_405, (4096, 1024), (1024, 1))
    assert_size_stride(div_67, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_409, (1024, 1024), (1024, 1))
    assert_size_stride(permute_421, (1024, 1024), (1024, 1))
    assert_size_stride(permute_426, (1024, 1024), (1024, 1))
    assert_size_stride(permute_430, (1024, 1024), (1024, 1))
    assert_size_stride(div_69, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_434, (1024, 4096), (4096, 1))
    assert_size_stride(permute_438, (4096, 1024), (1024, 1))
    assert_size_stride(div_70, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_442, (1024, 1024), (1024, 1))
    assert_size_stride(permute_454, (1024, 1024), (1024, 1))
    assert_size_stride(permute_459, (1024, 1024), (1024, 1))
    assert_size_stride(permute_463, (1024, 1024), (1024, 1))
    assert_size_stride(div_72, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_467, (1024, 4096), (4096, 1))
    assert_size_stride(permute_471, (4096, 1024), (1024, 1))
    assert_size_stride(div_73, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_475, (1024, 1024), (1024, 1))
    assert_size_stride(permute_487, (1024, 1024), (1024, 1))
    assert_size_stride(permute_492, (1024, 1024), (1024, 1))
    assert_size_stride(permute_496, (1024, 1024), (1024, 1))
    assert_size_stride(div_75, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_500, (1024, 4096), (4096, 1))
    assert_size_stride(permute_504, (4096, 1024), (1024, 1))
    assert_size_stride(div_76, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_508, (1024, 1024), (1024, 1))
    assert_size_stride(permute_520, (1024, 1024), (1024, 1))
    assert_size_stride(permute_525, (1024, 1024), (1024, 1))
    assert_size_stride(permute_529, (1024, 1024), (1024, 1))
    assert_size_stride(div_78, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_533, (1024, 4096), (4096, 1))
    assert_size_stride(permute_537, (4096, 1024), (1024, 1))
    assert_size_stride(div_79, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_541, (1024, 1024), (1024, 1))
    assert_size_stride(permute_553, (1024, 1024), (1024, 1))
    assert_size_stride(permute_558, (1024, 1024), (1024, 1))
    assert_size_stride(permute_562, (1024, 1024), (1024, 1))
    assert_size_stride(div_81, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_566, (1024, 4096), (4096, 1))
    assert_size_stride(permute_570, (4096, 1024), (1024, 1))
    assert_size_stride(div_82, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_574, (1024, 1024), (1024, 1))
    assert_size_stride(permute_586, (1024, 1024), (1024, 1))
    assert_size_stride(permute_591, (1024, 1024), (1024, 1))
    assert_size_stride(permute_595, (1024, 1024), (1024, 1))
    assert_size_stride(div_84, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_599, (1024, 4096), (4096, 1))
    assert_size_stride(permute_603, (4096, 1024), (1024, 1))
    assert_size_stride(div_85, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_607, (1024, 1024), (1024, 1))
    assert_size_stride(permute_619, (1024, 1024), (1024, 1))
    assert_size_stride(permute_624, (1024, 1024), (1024, 1))
    assert_size_stride(permute_628, (1024, 1024), (1024, 1))
    assert_size_stride(div_87, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_632, (1024, 4096), (4096, 1))
    assert_size_stride(permute_636, (4096, 1024), (1024, 1))
    assert_size_stride(div_88, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_640, (1024, 1024), (1024, 1))
    assert_size_stride(permute_652, (1024, 1024), (1024, 1))
    assert_size_stride(permute_657, (1024, 1024), (1024, 1))
    assert_size_stride(permute_661, (1024, 1024), (1024, 1))
    assert_size_stride(div_90, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_665, (1024, 4096), (4096, 1))
    assert_size_stride(permute_669, (4096, 1024), (1024, 1))
    assert_size_stride(div_91, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_673, (1024, 1024), (1024, 1))
    assert_size_stride(permute_685, (1024, 1024), (1024, 1))
    assert_size_stride(permute_690, (1024, 1024), (1024, 1))
    assert_size_stride(permute_694, (1024, 1024), (1024, 1))
    assert_size_stride(div_93, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_698, (1024, 4096), (4096, 1))
    assert_size_stride(permute_702, (4096, 1024), (1024, 1))
    assert_size_stride(div_94, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_706, (1024, 1024), (1024, 1))
    assert_size_stride(permute_718, (1024, 1024), (1024, 1))
    assert_size_stride(permute_723, (1024, 1024), (1024, 1))
    assert_size_stride(permute_727, (1024, 1024), (1024, 1))
    assert_size_stride(div_96, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_731, (1024, 4096), (4096, 1))
    assert_size_stride(permute_735, (4096, 1024), (1024, 1))
    assert_size_stride(div_97, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_739, (1024, 1024), (1024, 1))
    assert_size_stride(permute_751, (1024, 1024), (1024, 1))
    assert_size_stride(permute_756, (1024, 1024), (1024, 1))
    assert_size_stride(permute_760, (1024, 1024), (1024, 1))
    assert_size_stride(div_99, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_764, (1024, 4096), (4096, 1))
    assert_size_stride(permute_768, (4096, 1024), (1024, 1))
    assert_size_stride(div_100, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_772, (1024, 1024), (1024, 1))
    assert_size_stride(permute_784, (1024, 1024), (1024, 1))
    assert_size_stride(permute_789, (1024, 1024), (1024, 1))
    assert_size_stride(permute_793, (1024, 1024), (1024, 1))
    assert_size_stride(div_102, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_797, (1024, 4096), (4096, 1))
    assert_size_stride(permute_801, (4096, 1024), (1024, 1))
    assert_size_stride(div_103, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_805, (1024, 1024), (1024, 1))
    assert_size_stride(permute_817, (1024, 1024), (1024, 1))
    assert_size_stride(permute_822, (1024, 1024), (1024, 1))
    assert_size_stride(permute_826, (1024, 1024), (1024, 1))
    assert_size_stride(div_105, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_830, (1024, 4096), (4096, 1))
    assert_size_stride(permute_834, (4096, 1024), (1024, 1))
    assert_size_stride(div_106, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_838, (1024, 1024), (1024, 1))
    assert_size_stride(permute_850, (1024, 1024), (1024, 1))
    assert_size_stride(permute_855, (1024, 1024), (1024, 1))
    assert_size_stride(permute_859, (1024, 1024), (1024, 1))
    assert_size_stride(div_108, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_863, (1024, 4096), (4096, 1))
    assert_size_stride(permute_867, (4096, 1024), (1024, 1))
    assert_size_stride(div_109, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_871, (1024, 1024), (1024, 1))
    assert_size_stride(permute_883, (1024, 1024), (1024, 1))
    assert_size_stride(permute_888, (1024, 1024), (1024, 1))
    assert_size_stride(permute_892, (1024, 1024), (1024, 1))
    assert_size_stride(div_111, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_896, (1024, 4096), (4096, 1))
    assert_size_stride(permute_900, (4096, 1024), (1024, 1))
    assert_size_stride(div_112, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_904, (1024, 1024), (1024, 1))
    assert_size_stride(permute_916, (1024, 1024), (1024, 1))
    assert_size_stride(permute_921, (1024, 1024), (1024, 1))
    assert_size_stride(permute_925, (1024, 1024), (1024, 1))
    assert_size_stride(div_114, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_929, (1024, 4096), (4096, 1))
    assert_size_stride(permute_933, (4096, 1024), (1024, 1))
    assert_size_stride(div_115, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_937, (1024, 1024), (1024, 1))
    assert_size_stride(permute_949, (1024, 1024), (1024, 1))
    assert_size_stride(permute_954, (1024, 1024), (1024, 1))
    assert_size_stride(permute_958, (1024, 1024), (1024, 1))
    assert_size_stride(div_117, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_962, (1024, 4096), (4096, 1))
    assert_size_stride(permute_966, (4096, 1024), (1024, 1))
    assert_size_stride(div_118, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_970, (1024, 1024), (1024, 1))
    assert_size_stride(permute_982, (1024, 1024), (1024, 1))
    assert_size_stride(permute_987, (1024, 1024), (1024, 1))
    assert_size_stride(permute_991, (1024, 1024), (1024, 1))
    assert_size_stride(div_120, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_995, (1024, 4096), (4096, 1))
    assert_size_stride(permute_999, (4096, 1024), (1024, 1))
    assert_size_stride(div_121, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_1003, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1015, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1020, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1024, (1024, 1024), (1024, 1))
    assert_size_stride(div_123, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_1028, (1024, 4096), (4096, 1))
    assert_size_stride(permute_1032, (4096, 1024), (1024, 1))
    assert_size_stride(div_124, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_1036, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1048, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1053, (1024, 1024), (1024, 1))
    assert_size_stride(permute_1057, (1024, 1024), (1024, 1))
    assert_size_stride(div_126, (1, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 512), (512, 1))
    assert_size_stride(tangents_3, (1, 512), (512, 1))
    buf0 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_nll_loss_backward_0(c_void_p(buf0.data_ptr()))
    aten.scatter_(buf0,1,where_4,-1.0)
    del where_4
    buf4 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_nll_loss_backward_1(c_void_p(buf4.data_ptr()))
    aten.scatter_(buf4,1,where_6,-1.0)
    del where_6
    buf3 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf7 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf8 = empty((1, 512, 2), device='cpu', dtype=torch.float32)
    cpp_fused__log_softmax_backward_data_cat_div_nll_loss_backward_nll_loss_forward_2(c_void_p(buf0.data_ptr()), c_void_p(ne_6.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(ne_3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(ne_8.data_ptr()), c_void_p(ne.data_ptr()), c_void_p(tangents_2.data_ptr()), c_void_p(sub_75.data_ptr()), c_void_p(tangents_3.data_ptr()), c_void_p(sub_77.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()))
    del buf3
    del buf7
    del ne
    del ne_3
    del ne_6
    del ne_8
    del sub_75
    del sub_77
    del tangents_1
    del tangents_2
    del tangents_3
    buf9 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf8, (512, 2), (2, 1), 0), permute_265, out=buf9)
    del permute_265
    buf10 = empty((2, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf8, (2, 512), (1, 2), 0), view_528, out=buf10)
    del view_528
    buf11 = empty((1, 2), device='cpu', dtype=torch.float32)
    buf12 = reinterpret_tensor(buf4, (1, 512, 1), (512, 1, 512), 0); del buf4  # reuse
    buf13 = reinterpret_tensor(buf0, (1, 512, 1), (512, 1, 512), 0); del buf0  # reuse
    buf14 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    buf15 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf16 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf17 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_native_dropout_backward_native_layer_norm_backward_sum_3(c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(primals_388.data_ptr()), c_void_p(mul_169.data_ptr()), c_void_p(div_54.data_ptr()), c_void_p(getitem_241.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()))
    del div_54
    del getitem_241
    del mul_169
    del primals_388
    buf18 = empty((512, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf17, (512, 1024), (1024, 1), 0), permute_269, out=buf18)
    del permute_269
    buf19 = empty((1024, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf17, (1024, 512), (1, 1024), 0), view_526, out=buf19)
    del view_526
    buf20 = reinterpret_tensor(buf8, (1, 1024), (1024, 1), 0); del buf8  # reuse
    buf21 = reinterpret_tensor(buf18, (1, 512, 4096), (2097152, 4096, 1), 0); del buf18  # reuse
    cpp_fused_gelu_gelu_backward_sum_4(c_void_p(buf21.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(addmm_142.data_ptr()), c_void_p(buf20.data_ptr()))
    del addmm_142
    buf22 = reinterpret_tensor(buf17, (512, 1024), (1024, 1), 0); del buf17  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf21, (512, 4096), (4096, 1), 0), permute_273, out=buf22)
    del permute_273
    buf23 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf21, (4096, 512), (1, 4096), 0), view_524, out=buf23)
    del view_524
    buf24 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf25 = buf13; del buf13  # reuse
    buf26 = buf12; del buf12  # reuse
    buf27 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf28 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf29 = buf14; del buf14  # reuse
    buf30 = reinterpret_tensor(buf9, (1, 512, 1024), (524288, 1024, 1), 0); del buf9  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_5(c_void_p(buf29.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(primals_382.data_ptr()), c_void_p(mul_164.data_ptr()), c_void_p(div_55.data_ptr()), c_void_p(getitem_237.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()))
    del div_55
    del getitem_237
    del mul_164
    del primals_382
    buf31 = buf22; del buf22  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf30, (512, 1024), (1024, 1), 0), permute_277, out=buf31)
    del permute_277
    buf32 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf30, (1024, 512), (1, 1024), 0), view_522, out=buf32)
    del view_522
    buf33 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_6(c_void_p(buf30.data_ptr()), c_void_p(buf33.data_ptr()))
    buf34 = reinterpret_tensor(buf30, (16, 512, 64), (32768, 64, 1), 0); del buf30  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_1, reinterpret_tensor(buf31, (16, 512, 64), (64, 1024, 1), 0), out=buf34)
    del permute_default_1
    buf35 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf31, (16, 512, 64), (64, 1024, 1), 0), permute_default_2, out=buf35)
    del permute_default_2
    buf36 = empty_strided((1, 16, 512, 1), (8192, 512, 1, 8192), device='cpu', dtype=torch.float32)
    buf37 = reinterpret_tensor(buf35, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf35  # reuse
    cpp_fused_7(c_void_p(buf37.data_ptr()), c_void_p(getitem_247.data_ptr()), c_void_p(alias_default_1.data_ptr()), c_void_p(buf36.data_ptr()))
    del alias_default_1
    del getitem_247
    buf38 = reinterpret_tensor(buf31, (16, 64, 512), (32768, 512, 1), 0); del buf31  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_3, reinterpret_tensor(buf37, (16, 512, 512), (262144, 512, 1), 0), out=buf38)
    del permute_default_3
    buf39 = empty((16, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf37, (16, 512, 512), (262144, 512, 1), 0), permute_default_4, out=buf39)
    del permute_default_4
    buf40 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_view_8(c_void_p(buf34.data_ptr()), c_void_p(buf40.data_ptr()))
    buf41 = reinterpret_tensor(buf34, (512, 1024), (1024, 1), 0); del buf34  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf40, permute_289, out=buf41)
    del permute_289
    buf42 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf40, (1024, 512), (1, 1024), 0), view_506, out=buf42)
    buf43 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf44 = reinterpret_tensor(buf38, (512, 1024), (1, 512), 0); del buf38  # reuse
    cpp_fused_sum_view_9(c_void_p(buf44.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf43.data_ptr()))
    buf45 = buf40; del buf40  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf44, permute_294, out=buf45)
    del permute_294
    buf46 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf44, (1024, 512), (512, 1), 0), view_506, out=buf46)
    buf47 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf48 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_view_10(c_void_p(buf44.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    buf49 = reinterpret_tensor(buf44, (512, 1024), (1024, 1), 0); del buf44  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf48, permute_298, out=buf49)
    del permute_298
    buf50 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf48, (1024, 512), (1, 1024), 0), view_506, out=buf50)
    del view_506
    buf51 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf52 = buf26; del buf26  # reuse
    buf53 = buf25; del buf25  # reuse
    buf54 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf55 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf56 = buf29; del buf29  # reuse
    buf57 = reinterpret_tensor(buf39, (1, 512, 1024), (524288, 1024, 1), 0); del buf39  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_11(c_void_p(buf56.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(primals_372.data_ptr()), c_void_p(mul_162.data_ptr()), c_void_p(div_57.data_ptr()), c_void_p(getitem_231.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()))
    del div_57
    del getitem_231
    del mul_162
    del primals_372
    buf58 = reinterpret_tensor(buf21, (512, 4096), (4096, 1), 0); del buf21  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf57, (512, 1024), (1024, 1), 0), permute_302, out=buf58)
    del permute_302
    buf59 = reinterpret_tensor(buf37, (1024, 4096), (4096, 1), 0); del buf37  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf57, (1024, 512), (1, 1024), 0), view_504, out=buf59)
    del view_504
    buf60 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf61 = reinterpret_tensor(buf58, (1, 512, 4096), (2097152, 4096, 1), 0); del buf58  # reuse
    cpp_fused_gelu_gelu_backward_sum_12(c_void_p(buf61.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(addmm_136.data_ptr()), c_void_p(buf60.data_ptr()))
    del addmm_136
    buf62 = reinterpret_tensor(buf57, (512, 1024), (1024, 1), 0); del buf57  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf61, (512, 4096), (4096, 1), 0), permute_306, out=buf62)
    del permute_306
    buf63 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf61, (4096, 512), (1, 4096), 0), view_502, out=buf63)
    del view_502
    buf64 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf65 = buf53; del buf53  # reuse
    buf66 = buf52; del buf52  # reuse
    buf67 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf68 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf69 = buf56; del buf56  # reuse
    buf70 = reinterpret_tensor(buf49, (1, 512, 1024), (524288, 1024, 1), 0); del buf49  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_13(c_void_p(buf69.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(primals_366.data_ptr()), c_void_p(mul_157.data_ptr()), c_void_p(div_58.data_ptr()), c_void_p(getitem_227.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf70.data_ptr()))
    del div_58
    del getitem_227
    del mul_157
    del primals_366
    buf71 = buf62; del buf62  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf70, (512, 1024), (1024, 1), 0), permute_310, out=buf71)
    del permute_310
    buf72 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf70, (1024, 512), (1, 1024), 0), view_500, out=buf72)
    del view_500
    buf73 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_14(c_void_p(buf70.data_ptr()), c_void_p(buf73.data_ptr()))
    buf74 = reinterpret_tensor(buf70, (16, 512, 64), (32768, 64, 1), 0); del buf70  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_7, reinterpret_tensor(buf71, (16, 512, 64), (64, 1024, 1), 0), out=buf74)
    del permute_default_7
    buf75 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf71, (16, 512, 64), (64, 1024, 1), 0), permute_default_8, out=buf75)
    del permute_default_8
    buf76 = buf36; del buf36  # reuse
    buf77 = reinterpret_tensor(buf75, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf75  # reuse
    cpp_fused_15(c_void_p(buf77.data_ptr()), c_void_p(getitem_249.data_ptr()), c_void_p(alias_default_3.data_ptr()), c_void_p(buf76.data_ptr()))
    del alias_default_3
    del getitem_249
    buf78 = reinterpret_tensor(buf71, (16, 64, 512), (32768, 512, 1), 0); del buf71  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_9, reinterpret_tensor(buf77, (16, 512, 512), (262144, 512, 1), 0), out=buf78)
    del permute_default_9
    buf79 = reinterpret_tensor(buf48, (16, 512, 64), (32768, 64, 1), 0); del buf48  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf77, (16, 512, 512), (262144, 512, 1), 0), permute_default_10, out=buf79)
    del permute_default_10
    buf80 = buf45; del buf45  # reuse
    cpp_fused_view_16(c_void_p(buf74.data_ptr()), c_void_p(buf80.data_ptr()))
    buf81 = reinterpret_tensor(buf74, (512, 1024), (1024, 1), 0); del buf74  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf80, permute_322, out=buf81)
    del permute_322
    buf82 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf80, (1024, 512), (1, 1024), 0), view_484, out=buf82)
    buf83 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf84 = reinterpret_tensor(buf78, (512, 1024), (1, 512), 0); del buf78  # reuse
    cpp_fused_sum_view_17(c_void_p(buf84.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf83.data_ptr()))
    buf85 = buf80; del buf80  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf84, permute_327, out=buf85)
    del permute_327
    buf86 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf84, (1024, 512), (512, 1), 0), view_484, out=buf86)
    buf87 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf88 = buf41; del buf41  # reuse
    cpp_fused_sum_view_18(c_void_p(buf84.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()))
    buf89 = reinterpret_tensor(buf84, (512, 1024), (1024, 1), 0); del buf84  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf88, permute_331, out=buf89)
    del permute_331
    buf90 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf88, (1024, 512), (1, 1024), 0), view_484, out=buf90)
    del view_484
    buf91 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf92 = buf66; del buf66  # reuse
    buf93 = buf65; del buf65  # reuse
    buf94 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf95 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf96 = buf69; del buf69  # reuse
    buf97 = reinterpret_tensor(buf79, (1, 512, 1024), (524288, 1024, 1), 0); del buf79  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_19(c_void_p(buf96.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(primals_356.data_ptr()), c_void_p(mul_155.data_ptr()), c_void_p(div_60.data_ptr()), c_void_p(getitem_221.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf97.data_ptr()))
    del div_60
    del getitem_221
    del mul_155
    del primals_356
    buf98 = reinterpret_tensor(buf61, (512, 4096), (4096, 1), 0); del buf61  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf97, (512, 1024), (1024, 1), 0), permute_335, out=buf98)
    del permute_335
    buf99 = reinterpret_tensor(buf77, (1024, 4096), (4096, 1), 0); del buf77  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf97, (1024, 512), (1, 1024), 0), view_482, out=buf99)
    del view_482
    buf100 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf101 = reinterpret_tensor(buf98, (1, 512, 4096), (2097152, 4096, 1), 0); del buf98  # reuse
    cpp_fused_gelu_gelu_backward_sum_20(c_void_p(buf101.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(addmm_130.data_ptr()), c_void_p(buf100.data_ptr()))
    del addmm_130
    buf102 = reinterpret_tensor(buf97, (512, 1024), (1024, 1), 0); del buf97  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf101, (512, 4096), (4096, 1), 0), permute_339, out=buf102)
    del permute_339
    buf103 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf101, (4096, 512), (1, 4096), 0), view_480, out=buf103)
    del view_480
    buf104 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf105 = buf93; del buf93  # reuse
    buf106 = buf92; del buf92  # reuse
    buf107 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf108 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf109 = reinterpret_tensor(buf102, (1, 512, 1024), (524288, 1024, 1), 0); del buf102  # reuse
    buf110 = reinterpret_tensor(buf89, (1, 512, 1024), (524288, 1024, 1), 0); del buf89  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_21(c_void_p(buf109.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(primals_350.data_ptr()), c_void_p(mul_150.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(div_61.data_ptr()), c_void_p(getitem_217.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf110.data_ptr()))
    del div_61
    del getitem_217
    del mul_150
    del primals_350
    buf111 = reinterpret_tensor(buf96, (512, 1024), (1024, 1), 0); del buf96  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf110, (512, 1024), (1024, 1), 0), permute_343, out=buf111)
    del permute_343
    buf112 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf110, (1024, 512), (1, 1024), 0), view_478, out=buf112)
    del view_478
    buf113 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_22(c_void_p(buf110.data_ptr()), c_void_p(buf113.data_ptr()))
    buf114 = reinterpret_tensor(buf110, (16, 512, 64), (32768, 64, 1), 0); del buf110  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_13, reinterpret_tensor(buf111, (16, 512, 64), (64, 1024, 1), 0), out=buf114)
    del permute_default_13
    buf115 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf111, (16, 512, 64), (64, 1024, 1), 0), permute_default_14, out=buf115)
    del permute_default_14
    buf116 = buf76; del buf76  # reuse
    buf117 = reinterpret_tensor(buf115, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf115  # reuse
    cpp_fused_23(c_void_p(buf117.data_ptr()), c_void_p(getitem_251.data_ptr()), c_void_p(alias_default_5.data_ptr()), c_void_p(buf116.data_ptr()))
    del alias_default_5
    del getitem_251
    buf118 = reinterpret_tensor(buf111, (16, 64, 512), (32768, 512, 1), 0); del buf111  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_15, reinterpret_tensor(buf117, (16, 512, 512), (262144, 512, 1), 0), out=buf118)
    del permute_default_15
    buf119 = reinterpret_tensor(buf88, (16, 512, 64), (32768, 64, 1), 0); del buf88  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf117, (16, 512, 512), (262144, 512, 1), 0), permute_default_16, out=buf119)
    del permute_default_16
    buf120 = buf85; del buf85  # reuse
    cpp_fused_view_24(c_void_p(buf114.data_ptr()), c_void_p(buf120.data_ptr()))
    buf121 = reinterpret_tensor(buf114, (512, 1024), (1024, 1), 0); del buf114  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf120, permute_355, out=buf121)
    del permute_355
    buf122 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf120, (1024, 512), (1, 1024), 0), view_462, out=buf122)
    buf123 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf124 = reinterpret_tensor(buf118, (512, 1024), (1, 512), 0); del buf118  # reuse
    cpp_fused_sum_view_25(c_void_p(buf124.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf123.data_ptr()))
    buf125 = buf120; del buf120  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf124, permute_360, out=buf125)
    del permute_360
    buf126 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf124, (1024, 512), (512, 1), 0), view_462, out=buf126)
    buf127 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf128 = buf81; del buf81  # reuse
    cpp_fused_sum_view_26(c_void_p(buf124.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()))
    buf129 = reinterpret_tensor(buf124, (512, 1024), (1024, 1), 0); del buf124  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf128, permute_364, out=buf129)
    del permute_364
    buf130 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf128, (1024, 512), (1, 1024), 0), view_462, out=buf130)
    del view_462
    buf131 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf132 = buf106; del buf106  # reuse
    buf133 = buf105; del buf105  # reuse
    buf134 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf135 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf136 = buf109; del buf109  # reuse
    buf137 = reinterpret_tensor(buf119, (1, 512, 1024), (524288, 1024, 1), 0); del buf119  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_27(c_void_p(buf136.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(primals_340.data_ptr()), c_void_p(mul_148.data_ptr()), c_void_p(div_63.data_ptr()), c_void_p(getitem_211.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf137.data_ptr()))
    del div_63
    del getitem_211
    del mul_148
    del primals_340
    buf138 = reinterpret_tensor(buf101, (512, 4096), (4096, 1), 0); del buf101  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf137, (512, 1024), (1024, 1), 0), permute_368, out=buf138)
    del permute_368
    buf139 = reinterpret_tensor(buf117, (1024, 4096), (4096, 1), 0); del buf117  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf137, (1024, 512), (1, 1024), 0), view_460, out=buf139)
    del view_460
    buf140 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf141 = reinterpret_tensor(buf138, (1, 512, 4096), (2097152, 4096, 1), 0); del buf138  # reuse
    cpp_fused_gelu_gelu_backward_sum_28(c_void_p(buf141.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(addmm_124.data_ptr()), c_void_p(buf140.data_ptr()))
    del addmm_124
    buf142 = reinterpret_tensor(buf137, (512, 1024), (1024, 1), 0); del buf137  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf141, (512, 4096), (4096, 1), 0), permute_372, out=buf142)
    del permute_372
    buf143 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf141, (4096, 512), (1, 4096), 0), view_458, out=buf143)
    del view_458
    buf144 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf145 = buf133; del buf133  # reuse
    buf146 = buf132; del buf132  # reuse
    buf147 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf148 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf149 = buf136; del buf136  # reuse
    buf150 = reinterpret_tensor(buf129, (1, 512, 1024), (524288, 1024, 1), 0); del buf129  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_29(c_void_p(buf149.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(mul_143.data_ptr()), c_void_p(div_64.data_ptr()), c_void_p(getitem_207.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf150.data_ptr()))
    del div_64
    del getitem_207
    del mul_143
    del primals_334
    buf151 = buf142; del buf142  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf150, (512, 1024), (1024, 1), 0), permute_376, out=buf151)
    del permute_376
    buf152 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf150, (1024, 512), (1, 1024), 0), view_456, out=buf152)
    del view_456
    buf153 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_30(c_void_p(buf150.data_ptr()), c_void_p(buf153.data_ptr()))
    buf154 = reinterpret_tensor(buf150, (16, 512, 64), (32768, 64, 1), 0); del buf150  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_19, reinterpret_tensor(buf151, (16, 512, 64), (64, 1024, 1), 0), out=buf154)
    del permute_default_19
    buf155 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf151, (16, 512, 64), (64, 1024, 1), 0), permute_default_20, out=buf155)
    del permute_default_20
    buf156 = buf116; del buf116  # reuse
    buf157 = reinterpret_tensor(buf155, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf155  # reuse
    cpp_fused_31(c_void_p(buf157.data_ptr()), c_void_p(getitem_253.data_ptr()), c_void_p(alias_default_7.data_ptr()), c_void_p(buf156.data_ptr()))
    del alias_default_7
    del getitem_253
    buf158 = reinterpret_tensor(buf151, (16, 64, 512), (32768, 512, 1), 0); del buf151  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_21, reinterpret_tensor(buf157, (16, 512, 512), (262144, 512, 1), 0), out=buf158)
    del permute_default_21
    buf159 = reinterpret_tensor(buf128, (16, 512, 64), (32768, 64, 1), 0); del buf128  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf157, (16, 512, 512), (262144, 512, 1), 0), permute_default_22, out=buf159)
    del permute_default_22
    buf160 = buf125; del buf125  # reuse
    cpp_fused_view_32(c_void_p(buf154.data_ptr()), c_void_p(buf160.data_ptr()))
    buf161 = reinterpret_tensor(buf154, (512, 1024), (1024, 1), 0); del buf154  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf160, permute_388, out=buf161)
    del permute_388
    buf162 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf160, (1024, 512), (1, 1024), 0), view_440, out=buf162)
    buf163 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf164 = reinterpret_tensor(buf158, (512, 1024), (1, 512), 0); del buf158  # reuse
    cpp_fused_sum_view_33(c_void_p(buf164.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf163.data_ptr()))
    buf165 = buf160; del buf160  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf164, permute_393, out=buf165)
    del permute_393
    buf166 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf164, (1024, 512), (512, 1), 0), view_440, out=buf166)
    buf167 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf168 = buf121; del buf121  # reuse
    cpp_fused_sum_view_34(c_void_p(buf164.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()))
    buf169 = reinterpret_tensor(buf164, (512, 1024), (1024, 1), 0); del buf164  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf168, permute_397, out=buf169)
    del permute_397
    buf170 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf168, (1024, 512), (1, 1024), 0), view_440, out=buf170)
    del view_440
    buf171 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf172 = buf146; del buf146  # reuse
    buf173 = buf145; del buf145  # reuse
    buf174 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf175 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf176 = buf149; del buf149  # reuse
    buf177 = reinterpret_tensor(buf159, (1, 512, 1024), (524288, 1024, 1), 0); del buf159  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_35(c_void_p(buf176.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(mul_141.data_ptr()), c_void_p(div_66.data_ptr()), c_void_p(getitem_201.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf177.data_ptr()))
    del div_66
    del getitem_201
    del mul_141
    del primals_324
    buf178 = reinterpret_tensor(buf141, (512, 4096), (4096, 1), 0); del buf141  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf177, (512, 1024), (1024, 1), 0), permute_401, out=buf178)
    del permute_401
    buf179 = reinterpret_tensor(buf157, (1024, 4096), (4096, 1), 0); del buf157  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf177, (1024, 512), (1, 1024), 0), view_438, out=buf179)
    del view_438
    buf180 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf181 = reinterpret_tensor(buf178, (1, 512, 4096), (2097152, 4096, 1), 0); del buf178  # reuse
    cpp_fused_gelu_gelu_backward_sum_36(c_void_p(buf181.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(addmm_118.data_ptr()), c_void_p(buf180.data_ptr()))
    del addmm_118
    buf182 = reinterpret_tensor(buf177, (512, 1024), (1024, 1), 0); del buf177  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf181, (512, 4096), (4096, 1), 0), permute_405, out=buf182)
    del permute_405
    buf183 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf181, (4096, 512), (1, 4096), 0), view_436, out=buf183)
    del view_436
    buf184 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf185 = buf173; del buf173  # reuse
    buf186 = buf172; del buf172  # reuse
    buf187 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf188 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf189 = buf176; del buf176  # reuse
    buf190 = reinterpret_tensor(buf169, (1, 512, 1024), (524288, 1024, 1), 0); del buf169  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_37(c_void_p(buf189.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(mul_136.data_ptr()), c_void_p(div_67.data_ptr()), c_void_p(getitem_197.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf190.data_ptr()))
    del div_67
    del getitem_197
    del mul_136
    del primals_318
    buf191 = buf182; del buf182  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf190, (512, 1024), (1024, 1), 0), permute_409, out=buf191)
    del permute_409
    buf192 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf190, (1024, 512), (1, 1024), 0), view_434, out=buf192)
    del view_434
    buf193 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_38(c_void_p(buf190.data_ptr()), c_void_p(buf193.data_ptr()))
    buf194 = reinterpret_tensor(buf190, (16, 512, 64), (32768, 64, 1), 0); del buf190  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_25, reinterpret_tensor(buf191, (16, 512, 64), (64, 1024, 1), 0), out=buf194)
    del permute_default_25
    buf195 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf191, (16, 512, 64), (64, 1024, 1), 0), permute_default_26, out=buf195)
    del permute_default_26
    buf196 = buf156; del buf156  # reuse
    buf197 = reinterpret_tensor(buf195, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf195  # reuse
    cpp_fused_39(c_void_p(buf197.data_ptr()), c_void_p(getitem_255.data_ptr()), c_void_p(alias_default_9.data_ptr()), c_void_p(buf196.data_ptr()))
    del alias_default_9
    del getitem_255
    buf198 = reinterpret_tensor(buf191, (16, 64, 512), (32768, 512, 1), 0); del buf191  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_27, reinterpret_tensor(buf197, (16, 512, 512), (262144, 512, 1), 0), out=buf198)
    del permute_default_27
    buf199 = reinterpret_tensor(buf168, (16, 512, 64), (32768, 64, 1), 0); del buf168  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf197, (16, 512, 512), (262144, 512, 1), 0), permute_default_28, out=buf199)
    del permute_default_28
    buf200 = buf165; del buf165  # reuse
    cpp_fused_view_40(c_void_p(buf194.data_ptr()), c_void_p(buf200.data_ptr()))
    buf201 = reinterpret_tensor(buf194, (512, 1024), (1024, 1), 0); del buf194  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf200, permute_421, out=buf201)
    del permute_421
    buf202 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf200, (1024, 512), (1, 1024), 0), view_418, out=buf202)
    buf203 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf204 = reinterpret_tensor(buf198, (512, 1024), (1, 512), 0); del buf198  # reuse
    cpp_fused_sum_view_41(c_void_p(buf204.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf203.data_ptr()))
    buf205 = buf200; del buf200  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf204, permute_426, out=buf205)
    del permute_426
    buf206 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf204, (1024, 512), (512, 1), 0), view_418, out=buf206)
    buf207 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf208 = buf161; del buf161  # reuse
    cpp_fused_sum_view_42(c_void_p(buf204.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()))
    buf209 = reinterpret_tensor(buf204, (512, 1024), (1024, 1), 0); del buf204  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf208, permute_430, out=buf209)
    del permute_430
    buf210 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf208, (1024, 512), (1, 1024), 0), view_418, out=buf210)
    del view_418
    buf211 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf212 = buf186; del buf186  # reuse
    buf213 = buf185; del buf185  # reuse
    buf214 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf215 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf216 = buf189; del buf189  # reuse
    buf217 = reinterpret_tensor(buf199, (1, 512, 1024), (524288, 1024, 1), 0); del buf199  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_43(c_void_p(buf216.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(mul_134.data_ptr()), c_void_p(div_69.data_ptr()), c_void_p(getitem_191.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf217.data_ptr()))
    del div_69
    del getitem_191
    del mul_134
    del primals_308
    buf218 = reinterpret_tensor(buf181, (512, 4096), (4096, 1), 0); del buf181  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf217, (512, 1024), (1024, 1), 0), permute_434, out=buf218)
    del permute_434
    buf219 = reinterpret_tensor(buf197, (1024, 4096), (4096, 1), 0); del buf197  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf217, (1024, 512), (1, 1024), 0), view_416, out=buf219)
    del view_416
    buf220 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf221 = reinterpret_tensor(buf218, (1, 512, 4096), (2097152, 4096, 1), 0); del buf218  # reuse
    cpp_fused_gelu_gelu_backward_sum_44(c_void_p(buf221.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(addmm_112.data_ptr()), c_void_p(buf220.data_ptr()))
    del addmm_112
    buf222 = reinterpret_tensor(buf217, (512, 1024), (1024, 1), 0); del buf217  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf221, (512, 4096), (4096, 1), 0), permute_438, out=buf222)
    del permute_438
    buf223 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf221, (4096, 512), (1, 4096), 0), view_414, out=buf223)
    del view_414
    buf224 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf225 = buf213; del buf213  # reuse
    buf226 = buf212; del buf212  # reuse
    buf227 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf228 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf229 = buf216; del buf216  # reuse
    buf230 = reinterpret_tensor(buf209, (1, 512, 1024), (524288, 1024, 1), 0); del buf209  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_45(c_void_p(buf229.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(mul_129.data_ptr()), c_void_p(div_70.data_ptr()), c_void_p(getitem_187.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf230.data_ptr()))
    del div_70
    del getitem_187
    del mul_129
    del primals_302
    buf231 = buf222; del buf222  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf230, (512, 1024), (1024, 1), 0), permute_442, out=buf231)
    del permute_442
    buf232 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf230, (1024, 512), (1, 1024), 0), view_412, out=buf232)
    del view_412
    buf233 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_46(c_void_p(buf230.data_ptr()), c_void_p(buf233.data_ptr()))
    buf234 = reinterpret_tensor(buf230, (16, 512, 64), (32768, 64, 1), 0); del buf230  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_31, reinterpret_tensor(buf231, (16, 512, 64), (64, 1024, 1), 0), out=buf234)
    del permute_default_31
    buf235 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf231, (16, 512, 64), (64, 1024, 1), 0), permute_default_32, out=buf235)
    del permute_default_32
    buf236 = buf196; del buf196  # reuse
    buf237 = reinterpret_tensor(buf235, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf235  # reuse
    cpp_fused_47(c_void_p(buf237.data_ptr()), c_void_p(getitem_257.data_ptr()), c_void_p(alias_default_11.data_ptr()), c_void_p(buf236.data_ptr()))
    del alias_default_11
    del getitem_257
    buf238 = reinterpret_tensor(buf231, (16, 64, 512), (32768, 512, 1), 0); del buf231  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_33, reinterpret_tensor(buf237, (16, 512, 512), (262144, 512, 1), 0), out=buf238)
    del permute_default_33
    buf239 = reinterpret_tensor(buf208, (16, 512, 64), (32768, 64, 1), 0); del buf208  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf237, (16, 512, 512), (262144, 512, 1), 0), permute_default_34, out=buf239)
    del permute_default_34
    buf240 = buf205; del buf205  # reuse
    cpp_fused_view_48(c_void_p(buf234.data_ptr()), c_void_p(buf240.data_ptr()))
    buf241 = reinterpret_tensor(buf234, (512, 1024), (1024, 1), 0); del buf234  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf240, permute_454, out=buf241)
    del permute_454
    buf242 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf240, (1024, 512), (1, 1024), 0), view_396, out=buf242)
    buf243 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf244 = reinterpret_tensor(buf238, (512, 1024), (1, 512), 0); del buf238  # reuse
    cpp_fused_sum_view_49(c_void_p(buf244.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf243.data_ptr()))
    buf245 = buf240; del buf240  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf244, permute_459, out=buf245)
    del permute_459
    buf246 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf244, (1024, 512), (512, 1), 0), view_396, out=buf246)
    buf247 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf248 = buf201; del buf201  # reuse
    cpp_fused_sum_view_50(c_void_p(buf244.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()))
    buf249 = reinterpret_tensor(buf244, (512, 1024), (1024, 1), 0); del buf244  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf248, permute_463, out=buf249)
    del permute_463
    buf250 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf248, (1024, 512), (1, 1024), 0), view_396, out=buf250)
    del view_396
    buf251 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf252 = buf226; del buf226  # reuse
    buf253 = buf225; del buf225  # reuse
    buf254 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf255 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf256 = buf229; del buf229  # reuse
    buf257 = reinterpret_tensor(buf239, (1, 512, 1024), (524288, 1024, 1), 0); del buf239  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_51(c_void_p(buf256.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(mul_127.data_ptr()), c_void_p(div_72.data_ptr()), c_void_p(getitem_181.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf257.data_ptr()))
    del div_72
    del getitem_181
    del mul_127
    del primals_292
    buf258 = reinterpret_tensor(buf221, (512, 4096), (4096, 1), 0); del buf221  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf257, (512, 1024), (1024, 1), 0), permute_467, out=buf258)
    del permute_467
    buf259 = reinterpret_tensor(buf237, (1024, 4096), (4096, 1), 0); del buf237  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf257, (1024, 512), (1, 1024), 0), view_394, out=buf259)
    del view_394
    buf260 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf261 = reinterpret_tensor(buf258, (1, 512, 4096), (2097152, 4096, 1), 0); del buf258  # reuse
    cpp_fused_gelu_gelu_backward_sum_52(c_void_p(buf261.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(addmm_106.data_ptr()), c_void_p(buf260.data_ptr()))
    del addmm_106
    buf262 = reinterpret_tensor(buf257, (512, 1024), (1024, 1), 0); del buf257  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf261, (512, 4096), (4096, 1), 0), permute_471, out=buf262)
    del permute_471
    buf263 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf261, (4096, 512), (1, 4096), 0), view_392, out=buf263)
    del view_392
    buf264 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf265 = buf253; del buf253  # reuse
    buf266 = buf252; del buf252  # reuse
    buf267 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf268 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf269 = buf256; del buf256  # reuse
    buf270 = reinterpret_tensor(buf249, (1, 512, 1024), (524288, 1024, 1), 0); del buf249  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_53(c_void_p(buf269.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(mul_122.data_ptr()), c_void_p(div_73.data_ptr()), c_void_p(getitem_177.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf270.data_ptr()))
    del div_73
    del getitem_177
    del mul_122
    del primals_286
    buf271 = buf262; del buf262  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf270, (512, 1024), (1024, 1), 0), permute_475, out=buf271)
    del permute_475
    buf272 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf270, (1024, 512), (1, 1024), 0), view_390, out=buf272)
    del view_390
    buf273 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_54(c_void_p(buf270.data_ptr()), c_void_p(buf273.data_ptr()))
    buf274 = reinterpret_tensor(buf270, (16, 512, 64), (32768, 64, 1), 0); del buf270  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_37, reinterpret_tensor(buf271, (16, 512, 64), (64, 1024, 1), 0), out=buf274)
    del permute_default_37
    buf275 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf271, (16, 512, 64), (64, 1024, 1), 0), permute_default_38, out=buf275)
    del permute_default_38
    buf276 = buf236; del buf236  # reuse
    buf277 = reinterpret_tensor(buf275, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf275  # reuse
    cpp_fused_55(c_void_p(buf277.data_ptr()), c_void_p(getitem_259.data_ptr()), c_void_p(alias_default_13.data_ptr()), c_void_p(buf276.data_ptr()))
    del alias_default_13
    del getitem_259
    buf278 = reinterpret_tensor(buf271, (16, 64, 512), (32768, 512, 1), 0); del buf271  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_39, reinterpret_tensor(buf277, (16, 512, 512), (262144, 512, 1), 0), out=buf278)
    del permute_default_39
    buf279 = reinterpret_tensor(buf248, (16, 512, 64), (32768, 64, 1), 0); del buf248  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf277, (16, 512, 512), (262144, 512, 1), 0), permute_default_40, out=buf279)
    del permute_default_40
    buf280 = buf245; del buf245  # reuse
    cpp_fused_view_56(c_void_p(buf274.data_ptr()), c_void_p(buf280.data_ptr()))
    buf281 = reinterpret_tensor(buf274, (512, 1024), (1024, 1), 0); del buf274  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf280, permute_487, out=buf281)
    del permute_487
    buf282 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf280, (1024, 512), (1, 1024), 0), view_374, out=buf282)
    buf283 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf284 = reinterpret_tensor(buf278, (512, 1024), (1, 512), 0); del buf278  # reuse
    cpp_fused_sum_view_57(c_void_p(buf284.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf283.data_ptr()))
    buf285 = buf280; del buf280  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf284, permute_492, out=buf285)
    del permute_492
    buf286 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf284, (1024, 512), (512, 1), 0), view_374, out=buf286)
    buf287 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf288 = buf241; del buf241  # reuse
    cpp_fused_sum_view_58(c_void_p(buf284.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()))
    buf289 = reinterpret_tensor(buf284, (512, 1024), (1024, 1), 0); del buf284  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf288, permute_496, out=buf289)
    del permute_496
    buf290 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf288, (1024, 512), (1, 1024), 0), view_374, out=buf290)
    del view_374
    buf291 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf292 = buf266; del buf266  # reuse
    buf293 = buf265; del buf265  # reuse
    buf294 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf295 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf296 = buf269; del buf269  # reuse
    buf297 = reinterpret_tensor(buf279, (1, 512, 1024), (524288, 1024, 1), 0); del buf279  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_59(c_void_p(buf296.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(mul_120.data_ptr()), c_void_p(div_75.data_ptr()), c_void_p(getitem_171.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf297.data_ptr()))
    del div_75
    del getitem_171
    del mul_120
    del primals_276
    buf298 = reinterpret_tensor(buf261, (512, 4096), (4096, 1), 0); del buf261  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (512, 1024), (1024, 1), 0), permute_500, out=buf298)
    del permute_500
    buf299 = reinterpret_tensor(buf277, (1024, 4096), (4096, 1), 0); del buf277  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf297, (1024, 512), (1, 1024), 0), view_372, out=buf299)
    del view_372
    buf300 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf301 = reinterpret_tensor(buf298, (1, 512, 4096), (2097152, 4096, 1), 0); del buf298  # reuse
    cpp_fused_gelu_gelu_backward_sum_60(c_void_p(buf301.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(addmm_100.data_ptr()), c_void_p(buf300.data_ptr()))
    del addmm_100
    buf302 = reinterpret_tensor(buf297, (512, 1024), (1024, 1), 0); del buf297  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf301, (512, 4096), (4096, 1), 0), permute_504, out=buf302)
    del permute_504
    buf303 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf301, (4096, 512), (1, 4096), 0), view_370, out=buf303)
    del view_370
    buf304 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf305 = buf293; del buf293  # reuse
    buf306 = buf292; del buf292  # reuse
    buf307 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf308 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf309 = buf296; del buf296  # reuse
    buf310 = reinterpret_tensor(buf289, (1, 512, 1024), (524288, 1024, 1), 0); del buf289  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_61(c_void_p(buf309.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(mul_115.data_ptr()), c_void_p(div_76.data_ptr()), c_void_p(getitem_167.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf310.data_ptr()))
    del div_76
    del getitem_167
    del mul_115
    del primals_270
    buf311 = buf302; del buf302  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf310, (512, 1024), (1024, 1), 0), permute_508, out=buf311)
    del permute_508
    buf312 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf310, (1024, 512), (1, 1024), 0), view_368, out=buf312)
    del view_368
    buf313 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_62(c_void_p(buf310.data_ptr()), c_void_p(buf313.data_ptr()))
    buf314 = reinterpret_tensor(buf310, (16, 512, 64), (32768, 64, 1), 0); del buf310  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_43, reinterpret_tensor(buf311, (16, 512, 64), (64, 1024, 1), 0), out=buf314)
    del permute_default_43
    buf315 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf311, (16, 512, 64), (64, 1024, 1), 0), permute_default_44, out=buf315)
    del permute_default_44
    buf316 = buf276; del buf276  # reuse
    buf317 = reinterpret_tensor(buf315, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf315  # reuse
    cpp_fused_63(c_void_p(buf317.data_ptr()), c_void_p(getitem_261.data_ptr()), c_void_p(alias_default_15.data_ptr()), c_void_p(buf316.data_ptr()))
    del alias_default_15
    del getitem_261
    buf318 = reinterpret_tensor(buf311, (16, 64, 512), (32768, 512, 1), 0); del buf311  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_45, reinterpret_tensor(buf317, (16, 512, 512), (262144, 512, 1), 0), out=buf318)
    del permute_default_45
    buf319 = reinterpret_tensor(buf288, (16, 512, 64), (32768, 64, 1), 0); del buf288  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf317, (16, 512, 512), (262144, 512, 1), 0), permute_default_46, out=buf319)
    del permute_default_46
    buf320 = buf285; del buf285  # reuse
    cpp_fused_view_64(c_void_p(buf314.data_ptr()), c_void_p(buf320.data_ptr()))
    buf321 = reinterpret_tensor(buf314, (512, 1024), (1024, 1), 0); del buf314  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf320, permute_520, out=buf321)
    del permute_520
    buf322 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf320, (1024, 512), (1, 1024), 0), view_352, out=buf322)
    buf323 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf324 = reinterpret_tensor(buf318, (512, 1024), (1, 512), 0); del buf318  # reuse
    cpp_fused_sum_view_65(c_void_p(buf324.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf323.data_ptr()))
    buf325 = buf320; del buf320  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf324, permute_525, out=buf325)
    del permute_525
    buf326 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf324, (1024, 512), (512, 1), 0), view_352, out=buf326)
    buf327 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf328 = buf281; del buf281  # reuse
    cpp_fused_sum_view_66(c_void_p(buf324.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()))
    buf329 = reinterpret_tensor(buf324, (512, 1024), (1024, 1), 0); del buf324  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf328, permute_529, out=buf329)
    del permute_529
    buf330 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf328, (1024, 512), (1, 1024), 0), view_352, out=buf330)
    del view_352
    buf331 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf332 = buf306; del buf306  # reuse
    buf333 = buf305; del buf305  # reuse
    buf334 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf335 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf336 = buf309; del buf309  # reuse
    buf337 = reinterpret_tensor(buf319, (1, 512, 1024), (524288, 1024, 1), 0); del buf319  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_67(c_void_p(buf336.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(mul_113.data_ptr()), c_void_p(div_78.data_ptr()), c_void_p(getitem_161.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf337.data_ptr()))
    del div_78
    del getitem_161
    del mul_113
    del primals_260
    buf338 = reinterpret_tensor(buf301, (512, 4096), (4096, 1), 0); del buf301  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf337, (512, 1024), (1024, 1), 0), permute_533, out=buf338)
    del permute_533
    buf339 = reinterpret_tensor(buf317, (1024, 4096), (4096, 1), 0); del buf317  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf337, (1024, 512), (1, 1024), 0), view_350, out=buf339)
    del view_350
    buf340 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf341 = reinterpret_tensor(buf338, (1, 512, 4096), (2097152, 4096, 1), 0); del buf338  # reuse
    cpp_fused_gelu_gelu_backward_sum_68(c_void_p(buf341.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(addmm_94.data_ptr()), c_void_p(buf340.data_ptr()))
    del addmm_94
    buf342 = reinterpret_tensor(buf337, (512, 1024), (1024, 1), 0); del buf337  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf341, (512, 4096), (4096, 1), 0), permute_537, out=buf342)
    del permute_537
    buf343 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf341, (4096, 512), (1, 4096), 0), view_348, out=buf343)
    del view_348
    buf344 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf345 = buf333; del buf333  # reuse
    buf346 = buf332; del buf332  # reuse
    buf347 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf348 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf349 = buf336; del buf336  # reuse
    buf350 = reinterpret_tensor(buf329, (1, 512, 1024), (524288, 1024, 1), 0); del buf329  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_69(c_void_p(buf349.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(mul_108.data_ptr()), c_void_p(div_79.data_ptr()), c_void_p(getitem_157.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf350.data_ptr()))
    del div_79
    del getitem_157
    del mul_108
    del primals_254
    buf351 = buf342; del buf342  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf350, (512, 1024), (1024, 1), 0), permute_541, out=buf351)
    del permute_541
    buf352 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf350, (1024, 512), (1, 1024), 0), view_346, out=buf352)
    del view_346
    buf353 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_70(c_void_p(buf350.data_ptr()), c_void_p(buf353.data_ptr()))
    buf354 = reinterpret_tensor(buf350, (16, 512, 64), (32768, 64, 1), 0); del buf350  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_49, reinterpret_tensor(buf351, (16, 512, 64), (64, 1024, 1), 0), out=buf354)
    del permute_default_49
    buf355 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf351, (16, 512, 64), (64, 1024, 1), 0), permute_default_50, out=buf355)
    del permute_default_50
    buf356 = buf316; del buf316  # reuse
    buf357 = reinterpret_tensor(buf355, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf355  # reuse
    cpp_fused_71(c_void_p(buf357.data_ptr()), c_void_p(getitem_263.data_ptr()), c_void_p(alias_default_17.data_ptr()), c_void_p(buf356.data_ptr()))
    del alias_default_17
    del getitem_263
    buf358 = reinterpret_tensor(buf351, (16, 64, 512), (32768, 512, 1), 0); del buf351  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_51, reinterpret_tensor(buf357, (16, 512, 512), (262144, 512, 1), 0), out=buf358)
    del permute_default_51
    buf359 = reinterpret_tensor(buf328, (16, 512, 64), (32768, 64, 1), 0); del buf328  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf357, (16, 512, 512), (262144, 512, 1), 0), permute_default_52, out=buf359)
    del permute_default_52
    buf360 = buf325; del buf325  # reuse
    cpp_fused_view_72(c_void_p(buf354.data_ptr()), c_void_p(buf360.data_ptr()))
    buf361 = reinterpret_tensor(buf354, (512, 1024), (1024, 1), 0); del buf354  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf360, permute_553, out=buf361)
    del permute_553
    buf362 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf360, (1024, 512), (1, 1024), 0), view_330, out=buf362)
    buf363 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf364 = reinterpret_tensor(buf358, (512, 1024), (1, 512), 0); del buf358  # reuse
    cpp_fused_sum_view_73(c_void_p(buf364.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf363.data_ptr()))
    buf365 = buf360; del buf360  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf364, permute_558, out=buf365)
    del permute_558
    buf366 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf364, (1024, 512), (512, 1), 0), view_330, out=buf366)
    buf367 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf368 = buf321; del buf321  # reuse
    cpp_fused_sum_view_74(c_void_p(buf364.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()))
    buf369 = reinterpret_tensor(buf364, (512, 1024), (1024, 1), 0); del buf364  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf368, permute_562, out=buf369)
    del permute_562
    buf370 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf368, (1024, 512), (1, 1024), 0), view_330, out=buf370)
    del view_330
    buf371 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf372 = buf346; del buf346  # reuse
    buf373 = buf345; del buf345  # reuse
    buf374 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf375 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf376 = buf349; del buf349  # reuse
    buf377 = reinterpret_tensor(buf359, (1, 512, 1024), (524288, 1024, 1), 0); del buf359  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_75(c_void_p(buf376.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(mul_106.data_ptr()), c_void_p(div_81.data_ptr()), c_void_p(getitem_151.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf377.data_ptr()))
    del div_81
    del getitem_151
    del mul_106
    del primals_244
    buf378 = reinterpret_tensor(buf341, (512, 4096), (4096, 1), 0); del buf341  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf377, (512, 1024), (1024, 1), 0), permute_566, out=buf378)
    del permute_566
    buf379 = reinterpret_tensor(buf357, (1024, 4096), (4096, 1), 0); del buf357  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf377, (1024, 512), (1, 1024), 0), view_328, out=buf379)
    del view_328
    buf380 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf381 = reinterpret_tensor(buf378, (1, 512, 4096), (2097152, 4096, 1), 0); del buf378  # reuse
    cpp_fused_gelu_gelu_backward_sum_76(c_void_p(buf381.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(addmm_88.data_ptr()), c_void_p(buf380.data_ptr()))
    del addmm_88
    buf382 = reinterpret_tensor(buf377, (512, 1024), (1024, 1), 0); del buf377  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf381, (512, 4096), (4096, 1), 0), permute_570, out=buf382)
    del permute_570
    buf383 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf381, (4096, 512), (1, 4096), 0), view_326, out=buf383)
    del view_326
    buf384 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf385 = buf373; del buf373  # reuse
    buf386 = buf372; del buf372  # reuse
    buf387 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf388 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf389 = buf376; del buf376  # reuse
    buf390 = reinterpret_tensor(buf369, (1, 512, 1024), (524288, 1024, 1), 0); del buf369  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_77(c_void_p(buf389.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(mul_101.data_ptr()), c_void_p(div_82.data_ptr()), c_void_p(getitem_147.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf390.data_ptr()))
    del div_82
    del getitem_147
    del mul_101
    del primals_238
    buf391 = buf382; del buf382  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf390, (512, 1024), (1024, 1), 0), permute_574, out=buf391)
    del permute_574
    buf392 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf390, (1024, 512), (1, 1024), 0), view_324, out=buf392)
    del view_324
    buf393 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_78(c_void_p(buf390.data_ptr()), c_void_p(buf393.data_ptr()))
    buf394 = reinterpret_tensor(buf390, (16, 512, 64), (32768, 64, 1), 0); del buf390  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_55, reinterpret_tensor(buf391, (16, 512, 64), (64, 1024, 1), 0), out=buf394)
    del permute_default_55
    buf395 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf391, (16, 512, 64), (64, 1024, 1), 0), permute_default_56, out=buf395)
    del permute_default_56
    buf396 = buf356; del buf356  # reuse
    buf397 = reinterpret_tensor(buf395, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf395  # reuse
    cpp_fused_79(c_void_p(buf397.data_ptr()), c_void_p(getitem_265.data_ptr()), c_void_p(alias_default_19.data_ptr()), c_void_p(buf396.data_ptr()))
    del alias_default_19
    del getitem_265
    buf398 = reinterpret_tensor(buf391, (16, 64, 512), (32768, 512, 1), 0); del buf391  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_57, reinterpret_tensor(buf397, (16, 512, 512), (262144, 512, 1), 0), out=buf398)
    del permute_default_57
    buf399 = reinterpret_tensor(buf368, (16, 512, 64), (32768, 64, 1), 0); del buf368  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf397, (16, 512, 512), (262144, 512, 1), 0), permute_default_58, out=buf399)
    del permute_default_58
    buf400 = buf365; del buf365  # reuse
    cpp_fused_view_80(c_void_p(buf394.data_ptr()), c_void_p(buf400.data_ptr()))
    buf401 = reinterpret_tensor(buf394, (512, 1024), (1024, 1), 0); del buf394  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf400, permute_586, out=buf401)
    del permute_586
    buf402 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf400, (1024, 512), (1, 1024), 0), view_308, out=buf402)
    buf403 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf404 = reinterpret_tensor(buf398, (512, 1024), (1, 512), 0); del buf398  # reuse
    cpp_fused_sum_view_81(c_void_p(buf404.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf403.data_ptr()))
    buf405 = buf400; del buf400  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf404, permute_591, out=buf405)
    del permute_591
    buf406 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf404, (1024, 512), (512, 1), 0), view_308, out=buf406)
    buf407 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf408 = buf361; del buf361  # reuse
    cpp_fused_sum_view_82(c_void_p(buf404.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf408.data_ptr()))
    buf409 = reinterpret_tensor(buf404, (512, 1024), (1024, 1), 0); del buf404  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf408, permute_595, out=buf409)
    del permute_595
    buf410 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf408, (1024, 512), (1, 1024), 0), view_308, out=buf410)
    del view_308
    buf411 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf412 = buf386; del buf386  # reuse
    buf413 = buf385; del buf385  # reuse
    buf414 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf415 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf416 = buf389; del buf389  # reuse
    buf417 = reinterpret_tensor(buf399, (1, 512, 1024), (524288, 1024, 1), 0); del buf399  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_83(c_void_p(buf416.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(mul_99.data_ptr()), c_void_p(div_84.data_ptr()), c_void_p(getitem_141.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf417.data_ptr()))
    del div_84
    del getitem_141
    del mul_99
    del primals_228
    buf418 = reinterpret_tensor(buf381, (512, 4096), (4096, 1), 0); del buf381  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf417, (512, 1024), (1024, 1), 0), permute_599, out=buf418)
    del permute_599
    buf419 = reinterpret_tensor(buf397, (1024, 4096), (4096, 1), 0); del buf397  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf417, (1024, 512), (1, 1024), 0), view_306, out=buf419)
    del view_306
    buf420 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf421 = reinterpret_tensor(buf418, (1, 512, 4096), (2097152, 4096, 1), 0); del buf418  # reuse
    cpp_fused_gelu_gelu_backward_sum_84(c_void_p(buf421.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(addmm_82.data_ptr()), c_void_p(buf420.data_ptr()))
    del addmm_82
    buf422 = reinterpret_tensor(buf417, (512, 1024), (1024, 1), 0); del buf417  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf421, (512, 4096), (4096, 1), 0), permute_603, out=buf422)
    del permute_603
    buf423 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf421, (4096, 512), (1, 4096), 0), view_304, out=buf423)
    del view_304
    buf424 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf425 = buf413; del buf413  # reuse
    buf426 = buf412; del buf412  # reuse
    buf427 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf428 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf429 = buf416; del buf416  # reuse
    buf430 = reinterpret_tensor(buf409, (1, 512, 1024), (524288, 1024, 1), 0); del buf409  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_85(c_void_p(buf429.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(mul_94.data_ptr()), c_void_p(div_85.data_ptr()), c_void_p(getitem_137.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf430.data_ptr()))
    del div_85
    del getitem_137
    del mul_94
    del primals_222
    buf431 = buf422; del buf422  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf430, (512, 1024), (1024, 1), 0), permute_607, out=buf431)
    del permute_607
    buf432 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf430, (1024, 512), (1, 1024), 0), view_302, out=buf432)
    del view_302
    buf433 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_86(c_void_p(buf430.data_ptr()), c_void_p(buf433.data_ptr()))
    buf434 = reinterpret_tensor(buf430, (16, 512, 64), (32768, 64, 1), 0); del buf430  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_61, reinterpret_tensor(buf431, (16, 512, 64), (64, 1024, 1), 0), out=buf434)
    del permute_default_61
    buf435 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf431, (16, 512, 64), (64, 1024, 1), 0), permute_default_62, out=buf435)
    del permute_default_62
    buf436 = buf396; del buf396  # reuse
    buf437 = reinterpret_tensor(buf435, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf435  # reuse
    cpp_fused_87(c_void_p(buf437.data_ptr()), c_void_p(getitem_267.data_ptr()), c_void_p(alias_default_21.data_ptr()), c_void_p(buf436.data_ptr()))
    del alias_default_21
    del getitem_267
    buf438 = reinterpret_tensor(buf431, (16, 64, 512), (32768, 512, 1), 0); del buf431  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_63, reinterpret_tensor(buf437, (16, 512, 512), (262144, 512, 1), 0), out=buf438)
    del permute_default_63
    buf439 = reinterpret_tensor(buf408, (16, 512, 64), (32768, 64, 1), 0); del buf408  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf437, (16, 512, 512), (262144, 512, 1), 0), permute_default_64, out=buf439)
    del permute_default_64
    buf440 = buf405; del buf405  # reuse
    cpp_fused_view_88(c_void_p(buf434.data_ptr()), c_void_p(buf440.data_ptr()))
    buf441 = reinterpret_tensor(buf434, (512, 1024), (1024, 1), 0); del buf434  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf440, permute_619, out=buf441)
    del permute_619
    buf442 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf440, (1024, 512), (1, 1024), 0), view_286, out=buf442)
    buf443 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf444 = reinterpret_tensor(buf438, (512, 1024), (1, 512), 0); del buf438  # reuse
    cpp_fused_sum_view_89(c_void_p(buf444.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf443.data_ptr()))
    buf445 = buf440; del buf440  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf444, permute_624, out=buf445)
    del permute_624
    buf446 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf444, (1024, 512), (512, 1), 0), view_286, out=buf446)
    buf447 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf448 = buf401; del buf401  # reuse
    cpp_fused_sum_view_90(c_void_p(buf444.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()))
    buf449 = reinterpret_tensor(buf444, (512, 1024), (1024, 1), 0); del buf444  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf448, permute_628, out=buf449)
    del permute_628
    buf450 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf448, (1024, 512), (1, 1024), 0), view_286, out=buf450)
    del view_286
    buf451 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf452 = buf426; del buf426  # reuse
    buf453 = buf425; del buf425  # reuse
    buf454 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf455 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf456 = buf429; del buf429  # reuse
    buf457 = reinterpret_tensor(buf439, (1, 512, 1024), (524288, 1024, 1), 0); del buf439  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_91(c_void_p(buf456.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(mul_92.data_ptr()), c_void_p(div_87.data_ptr()), c_void_p(getitem_131.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf457.data_ptr()))
    del div_87
    del getitem_131
    del mul_92
    del primals_212
    buf458 = reinterpret_tensor(buf421, (512, 4096), (4096, 1), 0); del buf421  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf457, (512, 1024), (1024, 1), 0), permute_632, out=buf458)
    del permute_632
    buf459 = reinterpret_tensor(buf437, (1024, 4096), (4096, 1), 0); del buf437  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf457, (1024, 512), (1, 1024), 0), view_284, out=buf459)
    del view_284
    buf460 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf461 = reinterpret_tensor(buf458, (1, 512, 4096), (2097152, 4096, 1), 0); del buf458  # reuse
    cpp_fused_gelu_gelu_backward_sum_92(c_void_p(buf461.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(addmm_76.data_ptr()), c_void_p(buf460.data_ptr()))
    del addmm_76
    buf462 = reinterpret_tensor(buf457, (512, 1024), (1024, 1), 0); del buf457  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf461, (512, 4096), (4096, 1), 0), permute_636, out=buf462)
    del permute_636
    buf463 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf461, (4096, 512), (1, 4096), 0), view_282, out=buf463)
    del view_282
    buf464 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf465 = buf453; del buf453  # reuse
    buf466 = buf452; del buf452  # reuse
    buf467 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf468 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf469 = buf456; del buf456  # reuse
    buf470 = reinterpret_tensor(buf449, (1, 512, 1024), (524288, 1024, 1), 0); del buf449  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_93(c_void_p(buf469.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(mul_87.data_ptr()), c_void_p(div_88.data_ptr()), c_void_p(getitem_127.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf470.data_ptr()))
    del div_88
    del getitem_127
    del mul_87
    del primals_206
    buf471 = buf462; del buf462  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf470, (512, 1024), (1024, 1), 0), permute_640, out=buf471)
    del permute_640
    buf472 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf470, (1024, 512), (1, 1024), 0), view_280, out=buf472)
    del view_280
    buf473 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_94(c_void_p(buf470.data_ptr()), c_void_p(buf473.data_ptr()))
    buf474 = reinterpret_tensor(buf470, (16, 512, 64), (32768, 64, 1), 0); del buf470  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_67, reinterpret_tensor(buf471, (16, 512, 64), (64, 1024, 1), 0), out=buf474)
    del permute_default_67
    buf475 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf471, (16, 512, 64), (64, 1024, 1), 0), permute_default_68, out=buf475)
    del permute_default_68
    buf476 = buf436; del buf436  # reuse
    buf477 = reinterpret_tensor(buf475, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf475  # reuse
    cpp_fused_95(c_void_p(buf477.data_ptr()), c_void_p(getitem_269.data_ptr()), c_void_p(alias_default_23.data_ptr()), c_void_p(buf476.data_ptr()))
    del alias_default_23
    del getitem_269
    buf478 = reinterpret_tensor(buf471, (16, 64, 512), (32768, 512, 1), 0); del buf471  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_69, reinterpret_tensor(buf477, (16, 512, 512), (262144, 512, 1), 0), out=buf478)
    del permute_default_69
    buf479 = reinterpret_tensor(buf448, (16, 512, 64), (32768, 64, 1), 0); del buf448  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf477, (16, 512, 512), (262144, 512, 1), 0), permute_default_70, out=buf479)
    del permute_default_70
    buf480 = buf445; del buf445  # reuse
    cpp_fused_view_96(c_void_p(buf474.data_ptr()), c_void_p(buf480.data_ptr()))
    buf481 = reinterpret_tensor(buf474, (512, 1024), (1024, 1), 0); del buf474  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf480, permute_652, out=buf481)
    del permute_652
    buf482 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf480, (1024, 512), (1, 1024), 0), view_264, out=buf482)
    buf483 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf484 = reinterpret_tensor(buf478, (512, 1024), (1, 512), 0); del buf478  # reuse
    cpp_fused_sum_view_97(c_void_p(buf484.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf483.data_ptr()))
    buf485 = buf480; del buf480  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf484, permute_657, out=buf485)
    del permute_657
    buf486 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf484, (1024, 512), (512, 1), 0), view_264, out=buf486)
    buf487 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf488 = buf441; del buf441  # reuse
    cpp_fused_sum_view_98(c_void_p(buf484.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf488.data_ptr()))
    buf489 = reinterpret_tensor(buf484, (512, 1024), (1024, 1), 0); del buf484  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf488, permute_661, out=buf489)
    del permute_661
    buf490 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf488, (1024, 512), (1, 1024), 0), view_264, out=buf490)
    del view_264
    buf491 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf492 = buf466; del buf466  # reuse
    buf493 = buf465; del buf465  # reuse
    buf494 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf495 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf496 = buf469; del buf469  # reuse
    buf497 = reinterpret_tensor(buf479, (1, 512, 1024), (524288, 1024, 1), 0); del buf479  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_99(c_void_p(buf496.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(mul_85.data_ptr()), c_void_p(div_90.data_ptr()), c_void_p(getitem_121.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf497.data_ptr()))
    del div_90
    del getitem_121
    del mul_85
    del primals_196
    buf498 = reinterpret_tensor(buf461, (512, 4096), (4096, 1), 0); del buf461  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf497, (512, 1024), (1024, 1), 0), permute_665, out=buf498)
    del permute_665
    buf499 = reinterpret_tensor(buf477, (1024, 4096), (4096, 1), 0); del buf477  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf497, (1024, 512), (1, 1024), 0), view_262, out=buf499)
    del view_262
    buf500 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf501 = reinterpret_tensor(buf498, (1, 512, 4096), (2097152, 4096, 1), 0); del buf498  # reuse
    cpp_fused_gelu_gelu_backward_sum_100(c_void_p(buf501.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(addmm_70.data_ptr()), c_void_p(buf500.data_ptr()))
    del addmm_70
    buf502 = reinterpret_tensor(buf497, (512, 1024), (1024, 1), 0); del buf497  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf501, (512, 4096), (4096, 1), 0), permute_669, out=buf502)
    del permute_669
    buf503 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf501, (4096, 512), (1, 4096), 0), view_260, out=buf503)
    del view_260
    buf504 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf505 = buf493; del buf493  # reuse
    buf506 = buf492; del buf492  # reuse
    buf507 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf508 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf509 = buf496; del buf496  # reuse
    buf510 = reinterpret_tensor(buf489, (1, 512, 1024), (524288, 1024, 1), 0); del buf489  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_101(c_void_p(buf509.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(mul_80.data_ptr()), c_void_p(div_91.data_ptr()), c_void_p(getitem_117.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf510.data_ptr()))
    del div_91
    del getitem_117
    del mul_80
    del primals_190
    buf511 = buf502; del buf502  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf510, (512, 1024), (1024, 1), 0), permute_673, out=buf511)
    del permute_673
    buf512 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf510, (1024, 512), (1, 1024), 0), view_258, out=buf512)
    del view_258
    buf513 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_102(c_void_p(buf510.data_ptr()), c_void_p(buf513.data_ptr()))
    buf514 = reinterpret_tensor(buf510, (16, 512, 64), (32768, 64, 1), 0); del buf510  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_73, reinterpret_tensor(buf511, (16, 512, 64), (64, 1024, 1), 0), out=buf514)
    del permute_default_73
    buf515 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf511, (16, 512, 64), (64, 1024, 1), 0), permute_default_74, out=buf515)
    del permute_default_74
    buf516 = buf476; del buf476  # reuse
    buf517 = reinterpret_tensor(buf515, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf515  # reuse
    cpp_fused_103(c_void_p(buf517.data_ptr()), c_void_p(getitem_271.data_ptr()), c_void_p(alias_default_25.data_ptr()), c_void_p(buf516.data_ptr()))
    del alias_default_25
    del getitem_271
    buf518 = reinterpret_tensor(buf511, (16, 64, 512), (32768, 512, 1), 0); del buf511  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_75, reinterpret_tensor(buf517, (16, 512, 512), (262144, 512, 1), 0), out=buf518)
    del permute_default_75
    buf519 = reinterpret_tensor(buf488, (16, 512, 64), (32768, 64, 1), 0); del buf488  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf517, (16, 512, 512), (262144, 512, 1), 0), permute_default_76, out=buf519)
    del permute_default_76
    buf520 = buf485; del buf485  # reuse
    cpp_fused_view_104(c_void_p(buf514.data_ptr()), c_void_p(buf520.data_ptr()))
    buf521 = reinterpret_tensor(buf514, (512, 1024), (1024, 1), 0); del buf514  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf520, permute_685, out=buf521)
    del permute_685
    buf522 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf520, (1024, 512), (1, 1024), 0), view_242, out=buf522)
    buf523 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf524 = reinterpret_tensor(buf518, (512, 1024), (1, 512), 0); del buf518  # reuse
    cpp_fused_sum_view_105(c_void_p(buf524.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf523.data_ptr()))
    buf525 = buf520; del buf520  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf524, permute_690, out=buf525)
    del permute_690
    buf526 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf524, (1024, 512), (512, 1), 0), view_242, out=buf526)
    buf527 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf528 = buf481; del buf481  # reuse
    cpp_fused_sum_view_106(c_void_p(buf524.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(buf528.data_ptr()))
    buf529 = reinterpret_tensor(buf524, (512, 1024), (1024, 1), 0); del buf524  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf528, permute_694, out=buf529)
    del permute_694
    buf530 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf528, (1024, 512), (1, 1024), 0), view_242, out=buf530)
    del view_242
    buf531 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf532 = buf506; del buf506  # reuse
    buf533 = buf505; del buf505  # reuse
    buf534 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf535 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf536 = buf509; del buf509  # reuse
    buf537 = reinterpret_tensor(buf519, (1, 512, 1024), (524288, 1024, 1), 0); del buf519  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_107(c_void_p(buf536.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(mul_78.data_ptr()), c_void_p(div_93.data_ptr()), c_void_p(getitem_111.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf537.data_ptr()))
    del div_93
    del getitem_111
    del mul_78
    del primals_180
    buf538 = reinterpret_tensor(buf501, (512, 4096), (4096, 1), 0); del buf501  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf537, (512, 1024), (1024, 1), 0), permute_698, out=buf538)
    del permute_698
    buf539 = reinterpret_tensor(buf517, (1024, 4096), (4096, 1), 0); del buf517  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf537, (1024, 512), (1, 1024), 0), view_240, out=buf539)
    del view_240
    buf540 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf541 = reinterpret_tensor(buf538, (1, 512, 4096), (2097152, 4096, 1), 0); del buf538  # reuse
    cpp_fused_gelu_gelu_backward_sum_108(c_void_p(buf541.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(addmm_64.data_ptr()), c_void_p(buf540.data_ptr()))
    del addmm_64
    buf542 = reinterpret_tensor(buf537, (512, 1024), (1024, 1), 0); del buf537  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf541, (512, 4096), (4096, 1), 0), permute_702, out=buf542)
    del permute_702
    buf543 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf541, (4096, 512), (1, 4096), 0), view_238, out=buf543)
    del view_238
    buf544 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf545 = buf533; del buf533  # reuse
    buf546 = buf532; del buf532  # reuse
    buf547 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf548 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf549 = buf536; del buf536  # reuse
    buf550 = reinterpret_tensor(buf529, (1, 512, 1024), (524288, 1024, 1), 0); del buf529  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_109(c_void_p(buf549.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(mul_73.data_ptr()), c_void_p(div_94.data_ptr()), c_void_p(getitem_107.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf550.data_ptr()))
    del div_94
    del getitem_107
    del mul_73
    del primals_174
    buf551 = buf542; del buf542  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf550, (512, 1024), (1024, 1), 0), permute_706, out=buf551)
    del permute_706
    buf552 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf550, (1024, 512), (1, 1024), 0), view_236, out=buf552)
    del view_236
    buf553 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_110(c_void_p(buf550.data_ptr()), c_void_p(buf553.data_ptr()))
    buf554 = reinterpret_tensor(buf550, (16, 512, 64), (32768, 64, 1), 0); del buf550  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_79, reinterpret_tensor(buf551, (16, 512, 64), (64, 1024, 1), 0), out=buf554)
    del permute_default_79
    buf555 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf551, (16, 512, 64), (64, 1024, 1), 0), permute_default_80, out=buf555)
    del permute_default_80
    buf556 = buf516; del buf516  # reuse
    buf557 = reinterpret_tensor(buf555, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf555  # reuse
    cpp_fused_111(c_void_p(buf557.data_ptr()), c_void_p(getitem_273.data_ptr()), c_void_p(alias_default_27.data_ptr()), c_void_p(buf556.data_ptr()))
    del alias_default_27
    del getitem_273
    buf558 = reinterpret_tensor(buf551, (16, 64, 512), (32768, 512, 1), 0); del buf551  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_81, reinterpret_tensor(buf557, (16, 512, 512), (262144, 512, 1), 0), out=buf558)
    del permute_default_81
    buf559 = reinterpret_tensor(buf528, (16, 512, 64), (32768, 64, 1), 0); del buf528  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf557, (16, 512, 512), (262144, 512, 1), 0), permute_default_82, out=buf559)
    del permute_default_82
    buf560 = buf525; del buf525  # reuse
    cpp_fused_view_112(c_void_p(buf554.data_ptr()), c_void_p(buf560.data_ptr()))
    buf561 = reinterpret_tensor(buf554, (512, 1024), (1024, 1), 0); del buf554  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf560, permute_718, out=buf561)
    del permute_718
    buf562 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf560, (1024, 512), (1, 1024), 0), view_220, out=buf562)
    buf563 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf564 = reinterpret_tensor(buf558, (512, 1024), (1, 512), 0); del buf558  # reuse
    cpp_fused_sum_view_113(c_void_p(buf564.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf563.data_ptr()))
    buf565 = buf560; del buf560  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf564, permute_723, out=buf565)
    del permute_723
    buf566 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf564, (1024, 512), (512, 1), 0), view_220, out=buf566)
    buf567 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf568 = buf521; del buf521  # reuse
    cpp_fused_sum_view_114(c_void_p(buf564.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf568.data_ptr()))
    buf569 = reinterpret_tensor(buf564, (512, 1024), (1024, 1), 0); del buf564  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf568, permute_727, out=buf569)
    del permute_727
    buf570 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf568, (1024, 512), (1, 1024), 0), view_220, out=buf570)
    del view_220
    buf571 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf572 = buf546; del buf546  # reuse
    buf573 = buf545; del buf545  # reuse
    buf574 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf575 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf576 = buf549; del buf549  # reuse
    buf577 = reinterpret_tensor(buf559, (1, 512, 1024), (524288, 1024, 1), 0); del buf559  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_115(c_void_p(buf576.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(mul_71.data_ptr()), c_void_p(div_96.data_ptr()), c_void_p(getitem_101.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(buf573.data_ptr()), c_void_p(buf574.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf577.data_ptr()))
    del div_96
    del getitem_101
    del mul_71
    del primals_164
    buf578 = reinterpret_tensor(buf541, (512, 4096), (4096, 1), 0); del buf541  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf577, (512, 1024), (1024, 1), 0), permute_731, out=buf578)
    del permute_731
    buf579 = reinterpret_tensor(buf557, (1024, 4096), (4096, 1), 0); del buf557  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf577, (1024, 512), (1, 1024), 0), view_218, out=buf579)
    del view_218
    buf580 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf581 = reinterpret_tensor(buf578, (1, 512, 4096), (2097152, 4096, 1), 0); del buf578  # reuse
    cpp_fused_gelu_gelu_backward_sum_116(c_void_p(buf581.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(addmm_58.data_ptr()), c_void_p(buf580.data_ptr()))
    del addmm_58
    buf582 = reinterpret_tensor(buf577, (512, 1024), (1024, 1), 0); del buf577  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf581, (512, 4096), (4096, 1), 0), permute_735, out=buf582)
    del permute_735
    buf583 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf581, (4096, 512), (1, 4096), 0), view_216, out=buf583)
    del view_216
    buf584 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf585 = buf573; del buf573  # reuse
    buf586 = buf572; del buf572  # reuse
    buf587 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf588 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf589 = buf576; del buf576  # reuse
    buf590 = reinterpret_tensor(buf569, (1, 512, 1024), (524288, 1024, 1), 0); del buf569  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_117(c_void_p(buf589.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf582.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(mul_66.data_ptr()), c_void_p(div_97.data_ptr()), c_void_p(getitem_97.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(buf590.data_ptr()))
    del div_97
    del getitem_97
    del mul_66
    del primals_158
    buf591 = buf582; del buf582  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf590, (512, 1024), (1024, 1), 0), permute_739, out=buf591)
    del permute_739
    buf592 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf590, (1024, 512), (1, 1024), 0), view_214, out=buf592)
    del view_214
    buf593 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_118(c_void_p(buf590.data_ptr()), c_void_p(buf593.data_ptr()))
    buf594 = reinterpret_tensor(buf590, (16, 512, 64), (32768, 64, 1), 0); del buf590  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_85, reinterpret_tensor(buf591, (16, 512, 64), (64, 1024, 1), 0), out=buf594)
    del permute_default_85
    buf595 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf591, (16, 512, 64), (64, 1024, 1), 0), permute_default_86, out=buf595)
    del permute_default_86
    buf596 = buf556; del buf556  # reuse
    buf597 = reinterpret_tensor(buf595, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf595  # reuse
    cpp_fused_119(c_void_p(buf597.data_ptr()), c_void_p(getitem_275.data_ptr()), c_void_p(alias_default_29.data_ptr()), c_void_p(buf596.data_ptr()))
    del alias_default_29
    del getitem_275
    buf598 = reinterpret_tensor(buf591, (16, 64, 512), (32768, 512, 1), 0); del buf591  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_87, reinterpret_tensor(buf597, (16, 512, 512), (262144, 512, 1), 0), out=buf598)
    del permute_default_87
    buf599 = reinterpret_tensor(buf568, (16, 512, 64), (32768, 64, 1), 0); del buf568  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf597, (16, 512, 512), (262144, 512, 1), 0), permute_default_88, out=buf599)
    del permute_default_88
    buf600 = buf565; del buf565  # reuse
    cpp_fused_view_120(c_void_p(buf594.data_ptr()), c_void_p(buf600.data_ptr()))
    buf601 = reinterpret_tensor(buf594, (512, 1024), (1024, 1), 0); del buf594  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf600, permute_751, out=buf601)
    del permute_751
    buf602 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf600, (1024, 512), (1, 1024), 0), view_198, out=buf602)
    buf603 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf604 = reinterpret_tensor(buf598, (512, 1024), (1, 512), 0); del buf598  # reuse
    cpp_fused_sum_view_121(c_void_p(buf604.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(buf603.data_ptr()))
    buf605 = buf600; del buf600  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf604, permute_756, out=buf605)
    del permute_756
    buf606 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf604, (1024, 512), (512, 1), 0), view_198, out=buf606)
    buf607 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf608 = buf561; del buf561  # reuse
    cpp_fused_sum_view_122(c_void_p(buf604.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(buf608.data_ptr()))
    buf609 = reinterpret_tensor(buf604, (512, 1024), (1024, 1), 0); del buf604  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf608, permute_760, out=buf609)
    del permute_760
    buf610 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf608, (1024, 512), (1, 1024), 0), view_198, out=buf610)
    del view_198
    buf611 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf612 = buf586; del buf586  # reuse
    buf613 = buf585; del buf585  # reuse
    buf614 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf615 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf616 = buf589; del buf589  # reuse
    buf617 = reinterpret_tensor(buf599, (1, 512, 1024), (524288, 1024, 1), 0); del buf599  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_123(c_void_p(buf616.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(mul_64.data_ptr()), c_void_p(div_99.data_ptr()), c_void_p(getitem_91.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf615.data_ptr()), c_void_p(buf617.data_ptr()))
    del div_99
    del getitem_91
    del mul_64
    del primals_148
    buf618 = reinterpret_tensor(buf581, (512, 4096), (4096, 1), 0); del buf581  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf617, (512, 1024), (1024, 1), 0), permute_764, out=buf618)
    del permute_764
    buf619 = reinterpret_tensor(buf597, (1024, 4096), (4096, 1), 0); del buf597  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf617, (1024, 512), (1, 1024), 0), view_196, out=buf619)
    del view_196
    buf620 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf621 = reinterpret_tensor(buf618, (1, 512, 4096), (2097152, 4096, 1), 0); del buf618  # reuse
    cpp_fused_gelu_gelu_backward_sum_124(c_void_p(buf621.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(addmm_52.data_ptr()), c_void_p(buf620.data_ptr()))
    del addmm_52
    buf622 = reinterpret_tensor(buf617, (512, 1024), (1024, 1), 0); del buf617  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf621, (512, 4096), (4096, 1), 0), permute_768, out=buf622)
    del permute_768
    buf623 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf621, (4096, 512), (1, 4096), 0), view_194, out=buf623)
    del view_194
    buf624 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf625 = buf613; del buf613  # reuse
    buf626 = buf612; del buf612  # reuse
    buf627 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf628 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf629 = buf616; del buf616  # reuse
    buf630 = reinterpret_tensor(buf609, (1, 512, 1024), (524288, 1024, 1), 0); del buf609  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_125(c_void_p(buf629.data_ptr()), c_void_p(buf621.data_ptr()), c_void_p(buf622.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(mul_59.data_ptr()), c_void_p(div_100.data_ptr()), c_void_p(getitem_87.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(buf626.data_ptr()), c_void_p(buf627.data_ptr()), c_void_p(buf628.data_ptr()), c_void_p(buf630.data_ptr()))
    del div_100
    del getitem_87
    del mul_59
    del primals_142
    buf631 = buf622; del buf622  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf630, (512, 1024), (1024, 1), 0), permute_772, out=buf631)
    del permute_772
    buf632 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf630, (1024, 512), (1, 1024), 0), view_192, out=buf632)
    del view_192
    buf633 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_126(c_void_p(buf630.data_ptr()), c_void_p(buf633.data_ptr()))
    buf634 = reinterpret_tensor(buf630, (16, 512, 64), (32768, 64, 1), 0); del buf630  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_91, reinterpret_tensor(buf631, (16, 512, 64), (64, 1024, 1), 0), out=buf634)
    del permute_default_91
    buf635 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf631, (16, 512, 64), (64, 1024, 1), 0), permute_default_92, out=buf635)
    del permute_default_92
    buf636 = buf596; del buf596  # reuse
    buf637 = reinterpret_tensor(buf635, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf635  # reuse
    cpp_fused_127(c_void_p(buf637.data_ptr()), c_void_p(getitem_277.data_ptr()), c_void_p(alias_default_31.data_ptr()), c_void_p(buf636.data_ptr()))
    del alias_default_31
    del getitem_277
    buf638 = reinterpret_tensor(buf631, (16, 64, 512), (32768, 512, 1), 0); del buf631  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_93, reinterpret_tensor(buf637, (16, 512, 512), (262144, 512, 1), 0), out=buf638)
    del permute_default_93
    buf639 = reinterpret_tensor(buf608, (16, 512, 64), (32768, 64, 1), 0); del buf608  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf637, (16, 512, 512), (262144, 512, 1), 0), permute_default_94, out=buf639)
    del permute_default_94
    buf640 = buf605; del buf605  # reuse
    cpp_fused_view_128(c_void_p(buf634.data_ptr()), c_void_p(buf640.data_ptr()))
    buf641 = reinterpret_tensor(buf634, (512, 1024), (1024, 1), 0); del buf634  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf640, permute_784, out=buf641)
    del permute_784
    buf642 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf640, (1024, 512), (1, 1024), 0), view_176, out=buf642)
    buf643 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf644 = reinterpret_tensor(buf638, (512, 1024), (1, 512), 0); del buf638  # reuse
    cpp_fused_sum_view_129(c_void_p(buf644.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(buf643.data_ptr()))
    buf645 = buf640; del buf640  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf644, permute_789, out=buf645)
    del permute_789
    buf646 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf644, (1024, 512), (512, 1), 0), view_176, out=buf646)
    buf647 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf648 = buf601; del buf601  # reuse
    cpp_fused_sum_view_130(c_void_p(buf644.data_ptr()), c_void_p(buf639.data_ptr()), c_void_p(buf647.data_ptr()), c_void_p(buf648.data_ptr()))
    buf649 = reinterpret_tensor(buf644, (512, 1024), (1024, 1), 0); del buf644  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf648, permute_793, out=buf649)
    del permute_793
    buf650 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf648, (1024, 512), (1, 1024), 0), view_176, out=buf650)
    del view_176
    buf651 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf652 = buf626; del buf626  # reuse
    buf653 = buf625; del buf625  # reuse
    buf654 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf655 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf656 = buf629; del buf629  # reuse
    buf657 = reinterpret_tensor(buf639, (1, 512, 1024), (524288, 1024, 1), 0); del buf639  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_131(c_void_p(buf656.data_ptr()), c_void_p(buf648.data_ptr()), c_void_p(buf641.data_ptr()), c_void_p(buf645.data_ptr()), c_void_p(buf649.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(mul_57.data_ptr()), c_void_p(div_102.data_ptr()), c_void_p(getitem_81.data_ptr()), c_void_p(buf651.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(buf653.data_ptr()), c_void_p(buf654.data_ptr()), c_void_p(buf655.data_ptr()), c_void_p(buf657.data_ptr()))
    del div_102
    del getitem_81
    del mul_57
    del primals_132
    buf658 = reinterpret_tensor(buf621, (512, 4096), (4096, 1), 0); del buf621  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf657, (512, 1024), (1024, 1), 0), permute_797, out=buf658)
    del permute_797
    buf659 = reinterpret_tensor(buf637, (1024, 4096), (4096, 1), 0); del buf637  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf657, (1024, 512), (1, 1024), 0), view_174, out=buf659)
    del view_174
    buf660 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf661 = reinterpret_tensor(buf658, (1, 512, 4096), (2097152, 4096, 1), 0); del buf658  # reuse
    cpp_fused_gelu_gelu_backward_sum_132(c_void_p(buf661.data_ptr()), c_void_p(buf657.data_ptr()), c_void_p(addmm_46.data_ptr()), c_void_p(buf660.data_ptr()))
    del addmm_46
    buf662 = reinterpret_tensor(buf657, (512, 1024), (1024, 1), 0); del buf657  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf661, (512, 4096), (4096, 1), 0), permute_801, out=buf662)
    del permute_801
    buf663 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf661, (4096, 512), (1, 4096), 0), view_172, out=buf663)
    del view_172
    buf664 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf665 = buf653; del buf653  # reuse
    buf666 = buf652; del buf652  # reuse
    buf667 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf668 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf669 = buf656; del buf656  # reuse
    buf670 = reinterpret_tensor(buf649, (1, 512, 1024), (524288, 1024, 1), 0); del buf649  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_133(c_void_p(buf669.data_ptr()), c_void_p(buf661.data_ptr()), c_void_p(buf662.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(mul_52.data_ptr()), c_void_p(div_103.data_ptr()), c_void_p(getitem_77.data_ptr()), c_void_p(buf664.data_ptr()), c_void_p(buf665.data_ptr()), c_void_p(buf666.data_ptr()), c_void_p(buf667.data_ptr()), c_void_p(buf668.data_ptr()), c_void_p(buf670.data_ptr()))
    del div_103
    del getitem_77
    del mul_52
    del primals_126
    buf671 = buf662; del buf662  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf670, (512, 1024), (1024, 1), 0), permute_805, out=buf671)
    del permute_805
    buf672 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf670, (1024, 512), (1, 1024), 0), view_170, out=buf672)
    del view_170
    buf673 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_134(c_void_p(buf670.data_ptr()), c_void_p(buf673.data_ptr()))
    buf674 = reinterpret_tensor(buf670, (16, 512, 64), (32768, 64, 1), 0); del buf670  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_97, reinterpret_tensor(buf671, (16, 512, 64), (64, 1024, 1), 0), out=buf674)
    del permute_default_97
    buf675 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf671, (16, 512, 64), (64, 1024, 1), 0), permute_default_98, out=buf675)
    del permute_default_98
    buf676 = buf636; del buf636  # reuse
    buf677 = reinterpret_tensor(buf675, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf675  # reuse
    cpp_fused_135(c_void_p(buf677.data_ptr()), c_void_p(getitem_279.data_ptr()), c_void_p(alias_default_33.data_ptr()), c_void_p(buf676.data_ptr()))
    del alias_default_33
    del getitem_279
    buf678 = reinterpret_tensor(buf671, (16, 64, 512), (32768, 512, 1), 0); del buf671  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_99, reinterpret_tensor(buf677, (16, 512, 512), (262144, 512, 1), 0), out=buf678)
    del permute_default_99
    buf679 = reinterpret_tensor(buf648, (16, 512, 64), (32768, 64, 1), 0); del buf648  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf677, (16, 512, 512), (262144, 512, 1), 0), permute_default_100, out=buf679)
    del permute_default_100
    buf680 = buf645; del buf645  # reuse
    cpp_fused_view_136(c_void_p(buf674.data_ptr()), c_void_p(buf680.data_ptr()))
    buf681 = reinterpret_tensor(buf674, (512, 1024), (1024, 1), 0); del buf674  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf680, permute_817, out=buf681)
    del permute_817
    buf682 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf680, (1024, 512), (1, 1024), 0), view_154, out=buf682)
    buf683 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf684 = reinterpret_tensor(buf678, (512, 1024), (1, 512), 0); del buf678  # reuse
    cpp_fused_sum_view_137(c_void_p(buf684.data_ptr()), c_void_p(buf680.data_ptr()), c_void_p(buf683.data_ptr()))
    buf685 = buf680; del buf680  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf684, permute_822, out=buf685)
    del permute_822
    buf686 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf684, (1024, 512), (512, 1), 0), view_154, out=buf686)
    buf687 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf688 = buf641; del buf641  # reuse
    cpp_fused_sum_view_138(c_void_p(buf684.data_ptr()), c_void_p(buf679.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(buf688.data_ptr()))
    buf689 = reinterpret_tensor(buf684, (512, 1024), (1024, 1), 0); del buf684  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf688, permute_826, out=buf689)
    del permute_826
    buf690 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf688, (1024, 512), (1, 1024), 0), view_154, out=buf690)
    del view_154
    buf691 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf692 = buf666; del buf666  # reuse
    buf693 = buf665; del buf665  # reuse
    buf694 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf695 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf696 = buf669; del buf669  # reuse
    buf697 = reinterpret_tensor(buf679, (1, 512, 1024), (524288, 1024, 1), 0); del buf679  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_139(c_void_p(buf696.data_ptr()), c_void_p(buf688.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(buf685.data_ptr()), c_void_p(buf689.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(mul_50.data_ptr()), c_void_p(div_105.data_ptr()), c_void_p(getitem_71.data_ptr()), c_void_p(buf691.data_ptr()), c_void_p(buf692.data_ptr()), c_void_p(buf693.data_ptr()), c_void_p(buf694.data_ptr()), c_void_p(buf695.data_ptr()), c_void_p(buf697.data_ptr()))
    del div_105
    del getitem_71
    del mul_50
    del primals_116
    buf698 = reinterpret_tensor(buf661, (512, 4096), (4096, 1), 0); del buf661  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf697, (512, 1024), (1024, 1), 0), permute_830, out=buf698)
    del permute_830
    buf699 = reinterpret_tensor(buf677, (1024, 4096), (4096, 1), 0); del buf677  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf697, (1024, 512), (1, 1024), 0), view_152, out=buf699)
    del view_152
    buf700 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf701 = reinterpret_tensor(buf698, (1, 512, 4096), (2097152, 4096, 1), 0); del buf698  # reuse
    cpp_fused_gelu_gelu_backward_sum_140(c_void_p(buf701.data_ptr()), c_void_p(buf697.data_ptr()), c_void_p(addmm_40.data_ptr()), c_void_p(buf700.data_ptr()))
    del addmm_40
    buf702 = reinterpret_tensor(buf697, (512, 1024), (1024, 1), 0); del buf697  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf701, (512, 4096), (4096, 1), 0), permute_834, out=buf702)
    del permute_834
    buf703 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf701, (4096, 512), (1, 4096), 0), view_150, out=buf703)
    del view_150
    buf704 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf705 = buf693; del buf693  # reuse
    buf706 = buf692; del buf692  # reuse
    buf707 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf708 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf709 = buf696; del buf696  # reuse
    buf710 = reinterpret_tensor(buf689, (1, 512, 1024), (524288, 1024, 1), 0); del buf689  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_141(c_void_p(buf709.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf702.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(mul_45.data_ptr()), c_void_p(div_106.data_ptr()), c_void_p(getitem_67.data_ptr()), c_void_p(buf704.data_ptr()), c_void_p(buf705.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(buf707.data_ptr()), c_void_p(buf708.data_ptr()), c_void_p(buf710.data_ptr()))
    del div_106
    del getitem_67
    del mul_45
    del primals_110
    buf711 = buf702; del buf702  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf710, (512, 1024), (1024, 1), 0), permute_838, out=buf711)
    del permute_838
    buf712 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf710, (1024, 512), (1, 1024), 0), view_148, out=buf712)
    del view_148
    buf713 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_142(c_void_p(buf710.data_ptr()), c_void_p(buf713.data_ptr()))
    buf714 = reinterpret_tensor(buf710, (16, 512, 64), (32768, 64, 1), 0); del buf710  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_103, reinterpret_tensor(buf711, (16, 512, 64), (64, 1024, 1), 0), out=buf714)
    del permute_default_103
    buf715 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf711, (16, 512, 64), (64, 1024, 1), 0), permute_default_104, out=buf715)
    del permute_default_104
    buf716 = buf676; del buf676  # reuse
    buf717 = reinterpret_tensor(buf715, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf715  # reuse
    cpp_fused_143(c_void_p(buf717.data_ptr()), c_void_p(getitem_281.data_ptr()), c_void_p(alias_default_35.data_ptr()), c_void_p(buf716.data_ptr()))
    del alias_default_35
    del getitem_281
    buf718 = reinterpret_tensor(buf711, (16, 64, 512), (32768, 512, 1), 0); del buf711  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_105, reinterpret_tensor(buf717, (16, 512, 512), (262144, 512, 1), 0), out=buf718)
    del permute_default_105
    buf719 = reinterpret_tensor(buf688, (16, 512, 64), (32768, 64, 1), 0); del buf688  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf717, (16, 512, 512), (262144, 512, 1), 0), permute_default_106, out=buf719)
    del permute_default_106
    buf720 = buf685; del buf685  # reuse
    cpp_fused_view_144(c_void_p(buf714.data_ptr()), c_void_p(buf720.data_ptr()))
    buf721 = reinterpret_tensor(buf714, (512, 1024), (1024, 1), 0); del buf714  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf720, permute_850, out=buf721)
    del permute_850
    buf722 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf720, (1024, 512), (1, 1024), 0), view_132, out=buf722)
    buf723 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf724 = reinterpret_tensor(buf718, (512, 1024), (1, 512), 0); del buf718  # reuse
    cpp_fused_sum_view_145(c_void_p(buf724.data_ptr()), c_void_p(buf720.data_ptr()), c_void_p(buf723.data_ptr()))
    buf725 = buf720; del buf720  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf724, permute_855, out=buf725)
    del permute_855
    buf726 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf724, (1024, 512), (512, 1), 0), view_132, out=buf726)
    buf727 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf728 = buf681; del buf681  # reuse
    cpp_fused_sum_view_146(c_void_p(buf724.data_ptr()), c_void_p(buf719.data_ptr()), c_void_p(buf727.data_ptr()), c_void_p(buf728.data_ptr()))
    buf729 = reinterpret_tensor(buf724, (512, 1024), (1024, 1), 0); del buf724  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf728, permute_859, out=buf729)
    del permute_859
    buf730 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf728, (1024, 512), (1, 1024), 0), view_132, out=buf730)
    del view_132
    buf731 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf732 = buf706; del buf706  # reuse
    buf733 = buf705; del buf705  # reuse
    buf734 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf735 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf736 = buf709; del buf709  # reuse
    buf737 = reinterpret_tensor(buf719, (1, 512, 1024), (524288, 1024, 1), 0); del buf719  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_147(c_void_p(buf736.data_ptr()), c_void_p(buf728.data_ptr()), c_void_p(buf721.data_ptr()), c_void_p(buf725.data_ptr()), c_void_p(buf729.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(mul_43.data_ptr()), c_void_p(div_108.data_ptr()), c_void_p(getitem_61.data_ptr()), c_void_p(buf731.data_ptr()), c_void_p(buf732.data_ptr()), c_void_p(buf733.data_ptr()), c_void_p(buf734.data_ptr()), c_void_p(buf735.data_ptr()), c_void_p(buf737.data_ptr()))
    del div_108
    del getitem_61
    del mul_43
    del primals_100
    buf738 = reinterpret_tensor(buf701, (512, 4096), (4096, 1), 0); del buf701  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf737, (512, 1024), (1024, 1), 0), permute_863, out=buf738)
    del permute_863
    buf739 = reinterpret_tensor(buf717, (1024, 4096), (4096, 1), 0); del buf717  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf737, (1024, 512), (1, 1024), 0), view_130, out=buf739)
    del view_130
    buf740 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf741 = reinterpret_tensor(buf738, (1, 512, 4096), (2097152, 4096, 1), 0); del buf738  # reuse
    cpp_fused_gelu_gelu_backward_sum_148(c_void_p(buf741.data_ptr()), c_void_p(buf737.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(buf740.data_ptr()))
    del addmm_34
    buf742 = reinterpret_tensor(buf737, (512, 1024), (1024, 1), 0); del buf737  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf741, (512, 4096), (4096, 1), 0), permute_867, out=buf742)
    del permute_867
    buf743 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf741, (4096, 512), (1, 4096), 0), view_128, out=buf743)
    del view_128
    buf744 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf745 = buf733; del buf733  # reuse
    buf746 = buf732; del buf732  # reuse
    buf747 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf748 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf749 = buf736; del buf736  # reuse
    buf750 = reinterpret_tensor(buf729, (1, 512, 1024), (524288, 1024, 1), 0); del buf729  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_149(c_void_p(buf749.data_ptr()), c_void_p(buf741.data_ptr()), c_void_p(buf742.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(mul_38.data_ptr()), c_void_p(div_109.data_ptr()), c_void_p(getitem_57.data_ptr()), c_void_p(buf744.data_ptr()), c_void_p(buf745.data_ptr()), c_void_p(buf746.data_ptr()), c_void_p(buf747.data_ptr()), c_void_p(buf748.data_ptr()), c_void_p(buf750.data_ptr()))
    del div_109
    del getitem_57
    del mul_38
    del primals_94
    buf751 = buf742; del buf742  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf750, (512, 1024), (1024, 1), 0), permute_871, out=buf751)
    del permute_871
    buf752 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf750, (1024, 512), (1, 1024), 0), view_126, out=buf752)
    del view_126
    buf753 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_150(c_void_p(buf750.data_ptr()), c_void_p(buf753.data_ptr()))
    buf754 = reinterpret_tensor(buf750, (16, 512, 64), (32768, 64, 1), 0); del buf750  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_109, reinterpret_tensor(buf751, (16, 512, 64), (64, 1024, 1), 0), out=buf754)
    del permute_default_109
    buf755 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf751, (16, 512, 64), (64, 1024, 1), 0), permute_default_110, out=buf755)
    del permute_default_110
    buf756 = buf716; del buf716  # reuse
    buf757 = reinterpret_tensor(buf755, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf755  # reuse
    cpp_fused_151(c_void_p(buf757.data_ptr()), c_void_p(getitem_283.data_ptr()), c_void_p(alias_default_37.data_ptr()), c_void_p(buf756.data_ptr()))
    del alias_default_37
    del getitem_283
    buf758 = reinterpret_tensor(buf751, (16, 64, 512), (32768, 512, 1), 0); del buf751  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_111, reinterpret_tensor(buf757, (16, 512, 512), (262144, 512, 1), 0), out=buf758)
    del permute_default_111
    buf759 = reinterpret_tensor(buf728, (16, 512, 64), (32768, 64, 1), 0); del buf728  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf757, (16, 512, 512), (262144, 512, 1), 0), permute_default_112, out=buf759)
    del permute_default_112
    buf760 = buf725; del buf725  # reuse
    cpp_fused_view_152(c_void_p(buf754.data_ptr()), c_void_p(buf760.data_ptr()))
    buf761 = reinterpret_tensor(buf754, (512, 1024), (1024, 1), 0); del buf754  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf760, permute_883, out=buf761)
    del permute_883
    buf762 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf760, (1024, 512), (1, 1024), 0), view_110, out=buf762)
    buf763 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf764 = reinterpret_tensor(buf758, (512, 1024), (1, 512), 0); del buf758  # reuse
    cpp_fused_sum_view_153(c_void_p(buf764.data_ptr()), c_void_p(buf760.data_ptr()), c_void_p(buf763.data_ptr()))
    buf765 = buf760; del buf760  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf764, permute_888, out=buf765)
    del permute_888
    buf766 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf764, (1024, 512), (512, 1), 0), view_110, out=buf766)
    buf767 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf768 = buf721; del buf721  # reuse
    cpp_fused_sum_view_154(c_void_p(buf764.data_ptr()), c_void_p(buf759.data_ptr()), c_void_p(buf767.data_ptr()), c_void_p(buf768.data_ptr()))
    buf769 = reinterpret_tensor(buf764, (512, 1024), (1024, 1), 0); del buf764  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf768, permute_892, out=buf769)
    del permute_892
    buf770 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf768, (1024, 512), (1, 1024), 0), view_110, out=buf770)
    del view_110
    buf771 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf772 = buf746; del buf746  # reuse
    buf773 = buf745; del buf745  # reuse
    buf774 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf775 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf776 = buf749; del buf749  # reuse
    buf777 = reinterpret_tensor(buf759, (1, 512, 1024), (524288, 1024, 1), 0); del buf759  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_155(c_void_p(buf776.data_ptr()), c_void_p(buf768.data_ptr()), c_void_p(buf761.data_ptr()), c_void_p(buf765.data_ptr()), c_void_p(buf769.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(mul_36.data_ptr()), c_void_p(div_111.data_ptr()), c_void_p(getitem_51.data_ptr()), c_void_p(buf771.data_ptr()), c_void_p(buf772.data_ptr()), c_void_p(buf773.data_ptr()), c_void_p(buf774.data_ptr()), c_void_p(buf775.data_ptr()), c_void_p(buf777.data_ptr()))
    del div_111
    del getitem_51
    del mul_36
    del primals_84
    buf778 = reinterpret_tensor(buf741, (512, 4096), (4096, 1), 0); del buf741  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf777, (512, 1024), (1024, 1), 0), permute_896, out=buf778)
    del permute_896
    buf779 = reinterpret_tensor(buf757, (1024, 4096), (4096, 1), 0); del buf757  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf777, (1024, 512), (1, 1024), 0), view_108, out=buf779)
    del view_108
    buf780 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf781 = reinterpret_tensor(buf778, (1, 512, 4096), (2097152, 4096, 1), 0); del buf778  # reuse
    cpp_fused_gelu_gelu_backward_sum_156(c_void_p(buf781.data_ptr()), c_void_p(buf777.data_ptr()), c_void_p(addmm_28.data_ptr()), c_void_p(buf780.data_ptr()))
    del addmm_28
    buf782 = reinterpret_tensor(buf777, (512, 1024), (1024, 1), 0); del buf777  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf781, (512, 4096), (4096, 1), 0), permute_900, out=buf782)
    del permute_900
    buf783 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf781, (4096, 512), (1, 4096), 0), view_106, out=buf783)
    del view_106
    buf784 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf785 = buf773; del buf773  # reuse
    buf786 = buf772; del buf772  # reuse
    buf787 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf788 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf789 = buf776; del buf776  # reuse
    buf790 = reinterpret_tensor(buf769, (1, 512, 1024), (524288, 1024, 1), 0); del buf769  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_157(c_void_p(buf789.data_ptr()), c_void_p(buf781.data_ptr()), c_void_p(buf782.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(mul_31.data_ptr()), c_void_p(div_112.data_ptr()), c_void_p(getitem_47.data_ptr()), c_void_p(buf784.data_ptr()), c_void_p(buf785.data_ptr()), c_void_p(buf786.data_ptr()), c_void_p(buf787.data_ptr()), c_void_p(buf788.data_ptr()), c_void_p(buf790.data_ptr()))
    del div_112
    del getitem_47
    del mul_31
    del primals_78
    buf791 = buf782; del buf782  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf790, (512, 1024), (1024, 1), 0), permute_904, out=buf791)
    del permute_904
    buf792 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf790, (1024, 512), (1, 1024), 0), view_104, out=buf792)
    del view_104
    buf793 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_158(c_void_p(buf790.data_ptr()), c_void_p(buf793.data_ptr()))
    buf794 = reinterpret_tensor(buf790, (16, 512, 64), (32768, 64, 1), 0); del buf790  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_115, reinterpret_tensor(buf791, (16, 512, 64), (64, 1024, 1), 0), out=buf794)
    del permute_default_115
    buf795 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf791, (16, 512, 64), (64, 1024, 1), 0), permute_default_116, out=buf795)
    del permute_default_116
    buf796 = buf756; del buf756  # reuse
    buf797 = reinterpret_tensor(buf795, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf795  # reuse
    cpp_fused_159(c_void_p(buf797.data_ptr()), c_void_p(getitem_285.data_ptr()), c_void_p(alias_default_39.data_ptr()), c_void_p(buf796.data_ptr()))
    del alias_default_39
    del getitem_285
    buf798 = reinterpret_tensor(buf791, (16, 64, 512), (32768, 512, 1), 0); del buf791  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_117, reinterpret_tensor(buf797, (16, 512, 512), (262144, 512, 1), 0), out=buf798)
    del permute_default_117
    buf799 = reinterpret_tensor(buf768, (16, 512, 64), (32768, 64, 1), 0); del buf768  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf797, (16, 512, 512), (262144, 512, 1), 0), permute_default_118, out=buf799)
    del permute_default_118
    buf800 = buf765; del buf765  # reuse
    cpp_fused_view_160(c_void_p(buf794.data_ptr()), c_void_p(buf800.data_ptr()))
    buf801 = reinterpret_tensor(buf794, (512, 1024), (1024, 1), 0); del buf794  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf800, permute_916, out=buf801)
    del permute_916
    buf802 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf800, (1024, 512), (1, 1024), 0), view_88, out=buf802)
    buf803 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf804 = reinterpret_tensor(buf798, (512, 1024), (1, 512), 0); del buf798  # reuse
    cpp_fused_sum_view_161(c_void_p(buf804.data_ptr()), c_void_p(buf800.data_ptr()), c_void_p(buf803.data_ptr()))
    buf805 = buf800; del buf800  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf804, permute_921, out=buf805)
    del permute_921
    buf806 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf804, (1024, 512), (512, 1), 0), view_88, out=buf806)
    buf807 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf808 = buf761; del buf761  # reuse
    cpp_fused_sum_view_162(c_void_p(buf804.data_ptr()), c_void_p(buf799.data_ptr()), c_void_p(buf807.data_ptr()), c_void_p(buf808.data_ptr()))
    buf809 = reinterpret_tensor(buf804, (512, 1024), (1024, 1), 0); del buf804  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf808, permute_925, out=buf809)
    del permute_925
    buf810 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf808, (1024, 512), (1, 1024), 0), view_88, out=buf810)
    del view_88
    buf811 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf812 = buf786; del buf786  # reuse
    buf813 = buf785; del buf785  # reuse
    buf814 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf815 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf816 = buf789; del buf789  # reuse
    buf817 = reinterpret_tensor(buf799, (1, 512, 1024), (524288, 1024, 1), 0); del buf799  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_163(c_void_p(buf816.data_ptr()), c_void_p(buf808.data_ptr()), c_void_p(buf801.data_ptr()), c_void_p(buf805.data_ptr()), c_void_p(buf809.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(mul_29.data_ptr()), c_void_p(div_114.data_ptr()), c_void_p(getitem_41.data_ptr()), c_void_p(buf811.data_ptr()), c_void_p(buf812.data_ptr()), c_void_p(buf813.data_ptr()), c_void_p(buf814.data_ptr()), c_void_p(buf815.data_ptr()), c_void_p(buf817.data_ptr()))
    del div_114
    del getitem_41
    del mul_29
    del primals_68
    buf818 = reinterpret_tensor(buf781, (512, 4096), (4096, 1), 0); del buf781  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf817, (512, 1024), (1024, 1), 0), permute_929, out=buf818)
    del permute_929
    buf819 = reinterpret_tensor(buf797, (1024, 4096), (4096, 1), 0); del buf797  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf817, (1024, 512), (1, 1024), 0), view_86, out=buf819)
    del view_86
    buf820 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf821 = reinterpret_tensor(buf818, (1, 512, 4096), (2097152, 4096, 1), 0); del buf818  # reuse
    cpp_fused_gelu_gelu_backward_sum_164(c_void_p(buf821.data_ptr()), c_void_p(buf817.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(buf820.data_ptr()))
    del addmm_22
    buf822 = reinterpret_tensor(buf817, (512, 1024), (1024, 1), 0); del buf817  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf821, (512, 4096), (4096, 1), 0), permute_933, out=buf822)
    del permute_933
    buf823 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf821, (4096, 512), (1, 4096), 0), view_84, out=buf823)
    del view_84
    buf824 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf825 = buf813; del buf813  # reuse
    buf826 = buf812; del buf812  # reuse
    buf827 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf828 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf829 = buf816; del buf816  # reuse
    buf830 = reinterpret_tensor(buf809, (1, 512, 1024), (524288, 1024, 1), 0); del buf809  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_165(c_void_p(buf829.data_ptr()), c_void_p(buf821.data_ptr()), c_void_p(buf822.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(mul_24.data_ptr()), c_void_p(div_115.data_ptr()), c_void_p(getitem_37.data_ptr()), c_void_p(buf824.data_ptr()), c_void_p(buf825.data_ptr()), c_void_p(buf826.data_ptr()), c_void_p(buf827.data_ptr()), c_void_p(buf828.data_ptr()), c_void_p(buf830.data_ptr()))
    del div_115
    del getitem_37
    del mul_24
    del primals_62
    buf831 = buf822; del buf822  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf830, (512, 1024), (1024, 1), 0), permute_937, out=buf831)
    del permute_937
    buf832 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf830, (1024, 512), (1, 1024), 0), view_82, out=buf832)
    del view_82
    buf833 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_166(c_void_p(buf830.data_ptr()), c_void_p(buf833.data_ptr()))
    buf834 = reinterpret_tensor(buf830, (16, 512, 64), (32768, 64, 1), 0); del buf830  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_121, reinterpret_tensor(buf831, (16, 512, 64), (64, 1024, 1), 0), out=buf834)
    del permute_default_121
    buf835 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf831, (16, 512, 64), (64, 1024, 1), 0), permute_default_122, out=buf835)
    del permute_default_122
    buf836 = buf796; del buf796  # reuse
    buf837 = reinterpret_tensor(buf835, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf835  # reuse
    cpp_fused_167(c_void_p(buf837.data_ptr()), c_void_p(getitem_287.data_ptr()), c_void_p(alias_default_41.data_ptr()), c_void_p(buf836.data_ptr()))
    del alias_default_41
    del getitem_287
    buf838 = reinterpret_tensor(buf831, (16, 64, 512), (32768, 512, 1), 0); del buf831  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_123, reinterpret_tensor(buf837, (16, 512, 512), (262144, 512, 1), 0), out=buf838)
    del permute_default_123
    buf839 = reinterpret_tensor(buf808, (16, 512, 64), (32768, 64, 1), 0); del buf808  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf837, (16, 512, 512), (262144, 512, 1), 0), permute_default_124, out=buf839)
    del permute_default_124
    buf840 = buf805; del buf805  # reuse
    cpp_fused_view_168(c_void_p(buf834.data_ptr()), c_void_p(buf840.data_ptr()))
    buf841 = reinterpret_tensor(buf834, (512, 1024), (1024, 1), 0); del buf834  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf840, permute_949, out=buf841)
    del permute_949
    buf842 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf840, (1024, 512), (1, 1024), 0), view_66, out=buf842)
    buf843 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf844 = reinterpret_tensor(buf838, (512, 1024), (1, 512), 0); del buf838  # reuse
    cpp_fused_sum_view_169(c_void_p(buf844.data_ptr()), c_void_p(buf840.data_ptr()), c_void_p(buf843.data_ptr()))
    buf845 = buf840; del buf840  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf844, permute_954, out=buf845)
    del permute_954
    buf846 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf844, (1024, 512), (512, 1), 0), view_66, out=buf846)
    buf847 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf848 = buf801; del buf801  # reuse
    cpp_fused_sum_view_170(c_void_p(buf844.data_ptr()), c_void_p(buf839.data_ptr()), c_void_p(buf847.data_ptr()), c_void_p(buf848.data_ptr()))
    buf849 = reinterpret_tensor(buf844, (512, 1024), (1024, 1), 0); del buf844  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf848, permute_958, out=buf849)
    del permute_958
    buf850 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf848, (1024, 512), (1, 1024), 0), view_66, out=buf850)
    del view_66
    buf851 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf852 = buf826; del buf826  # reuse
    buf853 = buf825; del buf825  # reuse
    buf854 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf855 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf856 = buf829; del buf829  # reuse
    buf857 = reinterpret_tensor(buf839, (1, 512, 1024), (524288, 1024, 1), 0); del buf839  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_171(c_void_p(buf856.data_ptr()), c_void_p(buf848.data_ptr()), c_void_p(buf841.data_ptr()), c_void_p(buf845.data_ptr()), c_void_p(buf849.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(mul_22.data_ptr()), c_void_p(div_117.data_ptr()), c_void_p(getitem_31.data_ptr()), c_void_p(buf851.data_ptr()), c_void_p(buf852.data_ptr()), c_void_p(buf853.data_ptr()), c_void_p(buf854.data_ptr()), c_void_p(buf855.data_ptr()), c_void_p(buf857.data_ptr()))
    del div_117
    del getitem_31
    del mul_22
    del primals_52
    buf858 = reinterpret_tensor(buf821, (512, 4096), (4096, 1), 0); del buf821  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf857, (512, 1024), (1024, 1), 0), permute_962, out=buf858)
    del permute_962
    buf859 = reinterpret_tensor(buf837, (1024, 4096), (4096, 1), 0); del buf837  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf857, (1024, 512), (1, 1024), 0), view_64, out=buf859)
    del view_64
    buf860 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf861 = reinterpret_tensor(buf858, (1, 512, 4096), (2097152, 4096, 1), 0); del buf858  # reuse
    cpp_fused_gelu_gelu_backward_sum_172(c_void_p(buf861.data_ptr()), c_void_p(buf857.data_ptr()), c_void_p(addmm_16.data_ptr()), c_void_p(buf860.data_ptr()))
    del addmm_16
    buf862 = reinterpret_tensor(buf857, (512, 1024), (1024, 1), 0); del buf857  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf861, (512, 4096), (4096, 1), 0), permute_966, out=buf862)
    del permute_966
    buf863 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf861, (4096, 512), (1, 4096), 0), view_62, out=buf863)
    del view_62
    buf864 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf865 = buf853; del buf853  # reuse
    buf866 = buf852; del buf852  # reuse
    buf867 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf868 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf869 = buf856; del buf856  # reuse
    buf870 = reinterpret_tensor(buf849, (1, 512, 1024), (524288, 1024, 1), 0); del buf849  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_173(c_void_p(buf869.data_ptr()), c_void_p(buf861.data_ptr()), c_void_p(buf862.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(mul_17.data_ptr()), c_void_p(div_118.data_ptr()), c_void_p(getitem_27.data_ptr()), c_void_p(buf864.data_ptr()), c_void_p(buf865.data_ptr()), c_void_p(buf866.data_ptr()), c_void_p(buf867.data_ptr()), c_void_p(buf868.data_ptr()), c_void_p(buf870.data_ptr()))
    del div_118
    del getitem_27
    del mul_17
    del primals_46
    buf871 = buf862; del buf862  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf870, (512, 1024), (1024, 1), 0), permute_970, out=buf871)
    del permute_970
    buf872 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf870, (1024, 512), (1, 1024), 0), view_60, out=buf872)
    del view_60
    buf873 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_174(c_void_p(buf870.data_ptr()), c_void_p(buf873.data_ptr()))
    buf874 = reinterpret_tensor(buf870, (16, 512, 64), (32768, 64, 1), 0); del buf870  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_127, reinterpret_tensor(buf871, (16, 512, 64), (64, 1024, 1), 0), out=buf874)
    del permute_default_127
    buf875 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf871, (16, 512, 64), (64, 1024, 1), 0), permute_default_128, out=buf875)
    del permute_default_128
    buf876 = buf836; del buf836  # reuse
    buf877 = reinterpret_tensor(buf875, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf875  # reuse
    cpp_fused_175(c_void_p(buf877.data_ptr()), c_void_p(getitem_289.data_ptr()), c_void_p(alias_default_43.data_ptr()), c_void_p(buf876.data_ptr()))
    del alias_default_43
    del getitem_289
    buf878 = reinterpret_tensor(buf871, (16, 64, 512), (32768, 512, 1), 0); del buf871  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_129, reinterpret_tensor(buf877, (16, 512, 512), (262144, 512, 1), 0), out=buf878)
    del permute_default_129
    buf879 = reinterpret_tensor(buf848, (16, 512, 64), (32768, 64, 1), 0); del buf848  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf877, (16, 512, 512), (262144, 512, 1), 0), permute_default_130, out=buf879)
    del permute_default_130
    buf880 = buf845; del buf845  # reuse
    cpp_fused_view_176(c_void_p(buf874.data_ptr()), c_void_p(buf880.data_ptr()))
    buf881 = reinterpret_tensor(buf874, (512, 1024), (1024, 1), 0); del buf874  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf880, permute_982, out=buf881)
    del permute_982
    buf882 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf880, (1024, 512), (1, 1024), 0), view_44, out=buf882)
    buf883 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf884 = reinterpret_tensor(buf878, (512, 1024), (1, 512), 0); del buf878  # reuse
    cpp_fused_sum_view_177(c_void_p(buf884.data_ptr()), c_void_p(buf880.data_ptr()), c_void_p(buf883.data_ptr()))
    buf885 = buf880; del buf880  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf884, permute_987, out=buf885)
    del permute_987
    buf886 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf884, (1024, 512), (512, 1), 0), view_44, out=buf886)
    buf887 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf888 = buf841; del buf841  # reuse
    cpp_fused_sum_view_178(c_void_p(buf884.data_ptr()), c_void_p(buf879.data_ptr()), c_void_p(buf887.data_ptr()), c_void_p(buf888.data_ptr()))
    buf889 = reinterpret_tensor(buf884, (512, 1024), (1024, 1), 0); del buf884  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf888, permute_991, out=buf889)
    del permute_991
    buf890 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf888, (1024, 512), (1, 1024), 0), view_44, out=buf890)
    del view_44
    buf891 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf892 = buf866; del buf866  # reuse
    buf893 = buf865; del buf865  # reuse
    buf894 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf895 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf896 = buf869; del buf869  # reuse
    buf897 = reinterpret_tensor(buf879, (1, 512, 1024), (524288, 1024, 1), 0); del buf879  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_179(c_void_p(buf896.data_ptr()), c_void_p(buf888.data_ptr()), c_void_p(buf881.data_ptr()), c_void_p(buf885.data_ptr()), c_void_p(buf889.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(mul_15.data_ptr()), c_void_p(div_120.data_ptr()), c_void_p(getitem_21.data_ptr()), c_void_p(buf891.data_ptr()), c_void_p(buf892.data_ptr()), c_void_p(buf893.data_ptr()), c_void_p(buf894.data_ptr()), c_void_p(buf895.data_ptr()), c_void_p(buf897.data_ptr()))
    del div_120
    del getitem_21
    del mul_15
    del primals_36
    buf898 = reinterpret_tensor(buf861, (512, 4096), (4096, 1), 0); del buf861  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf897, (512, 1024), (1024, 1), 0), permute_995, out=buf898)
    del permute_995
    buf899 = reinterpret_tensor(buf877, (1024, 4096), (4096, 1), 0); del buf877  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf897, (1024, 512), (1, 1024), 0), view_42, out=buf899)
    del view_42
    buf900 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf901 = reinterpret_tensor(buf898, (1, 512, 4096), (2097152, 4096, 1), 0); del buf898  # reuse
    cpp_fused_gelu_gelu_backward_sum_180(c_void_p(buf901.data_ptr()), c_void_p(buf897.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(buf900.data_ptr()))
    del addmm_10
    buf902 = reinterpret_tensor(buf897, (512, 1024), (1024, 1), 0); del buf897  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf901, (512, 4096), (4096, 1), 0), permute_999, out=buf902)
    del permute_999
    buf903 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf901, (4096, 512), (1, 4096), 0), view_40, out=buf903)
    del view_40
    buf904 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf905 = buf893; del buf893  # reuse
    buf906 = buf892; del buf892  # reuse
    buf907 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf908 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf909 = buf896; del buf896  # reuse
    buf910 = reinterpret_tensor(buf889, (1, 512, 1024), (524288, 1024, 1), 0); del buf889  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_181(c_void_p(buf909.data_ptr()), c_void_p(buf901.data_ptr()), c_void_p(buf902.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(mul_10.data_ptr()), c_void_p(div_121.data_ptr()), c_void_p(getitem_17.data_ptr()), c_void_p(buf904.data_ptr()), c_void_p(buf905.data_ptr()), c_void_p(buf906.data_ptr()), c_void_p(buf907.data_ptr()), c_void_p(buf908.data_ptr()), c_void_p(buf910.data_ptr()))
    del div_121
    del getitem_17
    del mul_10
    del primals_30
    buf911 = buf902; del buf902  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf910, (512, 1024), (1024, 1), 0), permute_1003, out=buf911)
    del permute_1003
    buf912 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf910, (1024, 512), (1, 1024), 0), view_38, out=buf912)
    del view_38
    buf913 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_182(c_void_p(buf910.data_ptr()), c_void_p(buf913.data_ptr()))
    buf914 = reinterpret_tensor(buf910, (16, 512, 64), (32768, 64, 1), 0); del buf910  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_133, reinterpret_tensor(buf911, (16, 512, 64), (64, 1024, 1), 0), out=buf914)
    del permute_default_133
    buf915 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf911, (16, 512, 64), (64, 1024, 1), 0), permute_default_134, out=buf915)
    del permute_default_134
    buf916 = buf876; del buf876  # reuse
    buf917 = reinterpret_tensor(buf915, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf915  # reuse
    cpp_fused_183(c_void_p(buf917.data_ptr()), c_void_p(getitem_291.data_ptr()), c_void_p(alias_default_45.data_ptr()), c_void_p(buf916.data_ptr()))
    del alias_default_45
    del getitem_291
    buf918 = reinterpret_tensor(buf911, (16, 64, 512), (32768, 512, 1), 0); del buf911  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_135, reinterpret_tensor(buf917, (16, 512, 512), (262144, 512, 1), 0), out=buf918)
    del permute_default_135
    buf919 = reinterpret_tensor(buf888, (16, 512, 64), (32768, 64, 1), 0); del buf888  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf917, (16, 512, 512), (262144, 512, 1), 0), permute_default_136, out=buf919)
    del permute_default_136
    buf920 = buf885; del buf885  # reuse
    cpp_fused_view_184(c_void_p(buf914.data_ptr()), c_void_p(buf920.data_ptr()))
    buf921 = reinterpret_tensor(buf914, (512, 1024), (1024, 1), 0); del buf914  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf920, permute_1015, out=buf921)
    del permute_1015
    buf922 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf920, (1024, 512), (1, 1024), 0), view_22, out=buf922)
    buf923 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf924 = reinterpret_tensor(buf918, (512, 1024), (1, 512), 0); del buf918  # reuse
    cpp_fused_sum_view_185(c_void_p(buf924.data_ptr()), c_void_p(buf920.data_ptr()), c_void_p(buf923.data_ptr()))
    buf925 = buf920; del buf920  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf924, permute_1020, out=buf925)
    del permute_1020
    buf926 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf924, (1024, 512), (512, 1), 0), view_22, out=buf926)
    buf927 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf928 = buf881; del buf881  # reuse
    cpp_fused_sum_view_186(c_void_p(buf924.data_ptr()), c_void_p(buf919.data_ptr()), c_void_p(buf927.data_ptr()), c_void_p(buf928.data_ptr()))
    buf929 = reinterpret_tensor(buf924, (512, 1024), (1024, 1), 0); del buf924  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf928, permute_1024, out=buf929)
    del permute_1024
    buf930 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf928, (1024, 512), (1, 1024), 0), view_22, out=buf930)
    del view_22
    buf931 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf932 = buf906; del buf906  # reuse
    buf933 = buf905; del buf905  # reuse
    buf934 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf935 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf936 = buf909; del buf909  # reuse
    buf937 = reinterpret_tensor(buf919, (1, 512, 1024), (524288, 1024, 1), 0); del buf919  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_187(c_void_p(buf936.data_ptr()), c_void_p(buf928.data_ptr()), c_void_p(buf921.data_ptr()), c_void_p(buf925.data_ptr()), c_void_p(buf929.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(mul_8.data_ptr()), c_void_p(div_123.data_ptr()), c_void_p(getitem_11.data_ptr()), c_void_p(buf931.data_ptr()), c_void_p(buf932.data_ptr()), c_void_p(buf933.data_ptr()), c_void_p(buf934.data_ptr()), c_void_p(buf935.data_ptr()), c_void_p(buf937.data_ptr()))
    del div_123
    del getitem_11
    del mul_8
    del primals_20
    buf938 = reinterpret_tensor(buf901, (512, 4096), (4096, 1), 0); del buf901  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf937, (512, 1024), (1024, 1), 0), permute_1028, out=buf938)
    del permute_1028
    buf939 = reinterpret_tensor(buf917, (1024, 4096), (4096, 1), 0); del buf917  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf937, (1024, 512), (1, 1024), 0), view_20, out=buf939)
    del view_20
    buf940 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf941 = reinterpret_tensor(buf938, (1, 512, 4096), (2097152, 4096, 1), 0); del buf938  # reuse
    cpp_fused_gelu_gelu_backward_sum_188(c_void_p(buf941.data_ptr()), c_void_p(buf937.data_ptr()), c_void_p(addmm_4.data_ptr()), c_void_p(buf940.data_ptr()))
    del addmm_4
    buf942 = reinterpret_tensor(buf937, (512, 1024), (1024, 1), 0); del buf937  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf941, (512, 4096), (4096, 1), 0), permute_1032, out=buf942)
    del permute_1032
    buf943 = empty((4096, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf941, (4096, 512), (1, 4096), 0), view_18, out=buf943)
    del view_18
    buf944 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf945 = buf933; del buf933  # reuse
    buf946 = buf932; del buf932  # reuse
    buf947 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf948 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf949 = buf936; del buf936  # reuse
    buf950 = reinterpret_tensor(buf929, (1, 512, 1024), (524288, 1024, 1), 0); del buf929  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_189(c_void_p(buf949.data_ptr()), c_void_p(buf941.data_ptr()), c_void_p(buf942.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(mul_3.data_ptr()), c_void_p(div_124.data_ptr()), c_void_p(getitem_7.data_ptr()), c_void_p(buf944.data_ptr()), c_void_p(buf945.data_ptr()), c_void_p(buf946.data_ptr()), c_void_p(buf947.data_ptr()), c_void_p(buf948.data_ptr()), c_void_p(buf950.data_ptr()))
    del buf941
    del div_124
    del getitem_7
    del mul_3
    del primals_14
    buf951 = buf942; del buf942  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf950, (512, 1024), (1024, 1), 0), permute_1036, out=buf951)
    del permute_1036
    buf952 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf950, (1024, 512), (1, 1024), 0), view_16, out=buf952)
    del view_16
    buf953 = empty((1, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_sum_190(c_void_p(buf950.data_ptr()), c_void_p(buf953.data_ptr()))
    buf954 = reinterpret_tensor(buf950, (16, 512, 64), (32768, 64, 1), 0); del buf950  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_139, reinterpret_tensor(buf951, (16, 512, 64), (64, 1024, 1), 0), out=buf954)
    del permute_default_139
    buf955 = empty((16, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf951, (16, 512, 64), (64, 1024, 1), 0), permute_default_140, out=buf955)
    del permute_default_140
    buf956 = buf916; del buf916  # reuse
    buf957 = reinterpret_tensor(buf955, (1, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf955  # reuse
    cpp_fused_191(c_void_p(buf957.data_ptr()), c_void_p(getitem_293.data_ptr()), c_void_p(alias_default_47.data_ptr()), c_void_p(buf956.data_ptr()))
    del alias_default_47
    del buf956
    del getitem_293
    buf958 = reinterpret_tensor(buf951, (16, 64, 512), (32768, 512, 1), 0); del buf951  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_141, reinterpret_tensor(buf957, (16, 512, 512), (262144, 512, 1), 0), out=buf958)
    del permute_default_141
    buf959 = reinterpret_tensor(buf928, (16, 512, 64), (32768, 64, 1), 0); del buf928  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf957, (16, 512, 512), (262144, 512, 1), 0), permute_default_142, out=buf959)
    del buf957
    del permute_default_142
    buf960 = buf925; del buf925  # reuse
    cpp_fused_view_192(c_void_p(buf954.data_ptr()), c_void_p(buf960.data_ptr()))
    buf961 = reinterpret_tensor(buf954, (512, 1024), (1024, 1), 0); del buf954  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf960, permute_1048, out=buf961)
    del permute_1048
    buf962 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf960, (1024, 512), (1, 1024), 0), view, out=buf962)
    buf963 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf964 = reinterpret_tensor(buf958, (512, 1024), (1, 512), 0); del buf958  # reuse
    cpp_fused_sum_view_193(c_void_p(buf964.data_ptr()), c_void_p(buf960.data_ptr()), c_void_p(buf963.data_ptr()))
    buf965 = buf960; del buf960  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf964, permute_1053, out=buf965)
    del permute_1053
    buf966 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf964, (1024, 512), (512, 1), 0), view, out=buf966)
    buf967 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf968 = buf921; del buf921  # reuse
    cpp_fused_sum_view_194(c_void_p(buf964.data_ptr()), c_void_p(buf959.data_ptr()), c_void_p(buf967.data_ptr()), c_void_p(buf968.data_ptr()))
    buf969 = reinterpret_tensor(buf964, (512, 1024), (1024, 1), 0); del buf964  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf968, permute_1057, out=buf969)
    del permute_1057
    buf970 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf968, (1024, 512), (1, 1024), 0), view, out=buf970)
    del view
    buf971 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf972 = buf946; del buf946  # reuse
    buf973 = buf945; del buf945  # reuse
    buf974 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf975 = empty((1024, ), device='cpu', dtype=torch.float32)
    buf976 = buf949; del buf949  # reuse
    buf978 = reinterpret_tensor(buf959, (1, 512, 1024), (524288, 1024, 1), 0); del buf959  # reuse
    buf986 = empty((1, 512, 1024), device='cpu', dtype=torch.float32)
    buf977 = empty((512, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_sum_195(c_void_p(buf976.data_ptr()), c_void_p(buf968.data_ptr()), c_void_p(buf961.data_ptr()), c_void_p(buf965.data_ptr()), c_void_p(buf969.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(mul_1.data_ptr()), c_void_p(div_126.data_ptr()), c_void_p(slice_3.data_ptr()), c_void_p(getitem_1.data_ptr()), c_void_p(primals_393.data_ptr()), c_void_p(buf971.data_ptr()), c_void_p(buf972.data_ptr()), c_void_p(buf973.data_ptr()), c_void_p(buf974.data_ptr()), c_void_p(buf975.data_ptr()), c_void_p(buf978.data_ptr()), c_void_p(buf986.data_ptr()), c_void_p(buf977.data_ptr()))
    del buf961
    del buf965
    del buf968
    del buf969
    del buf972
    del buf973
    del div_126
    del mul_1
    del primals_4
    aten.index_put_(buf977, [slice_3], buf978, True)
    del buf978
    del slice_3
    buf981 = empty((2, 1024), device='cpu', dtype=torch.float32)
    buf982 = buf976; del buf976  # reuse
    cpp_fused_embedding_dense_backward_native_dropout_backward_nll_loss_forward_196(c_void_p(buf982.data_ptr()), c_void_p(getitem_1.data_ptr()), c_void_p(buf981.data_ptr()))
    del getitem_1
    aten.index_put_(buf981, [full_default], buf982, True)
    del buf982
    del full_default
    buf985 = empty((29056, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_197(c_void_p(buf985.data_ptr()))
    aten.index_put_(buf985, [primals_393], buf986, True)
    del buf986
    del primals_393
    return (buf985, buf981, buf977, buf974, buf975, reinterpret_tensor(buf970, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf971, (1024, ), (1, ), 0), reinterpret_tensor(buf966, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf967, (1024, ), (1, ), 0), reinterpret_tensor(buf962, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf963, (1024, ), (1, ), 0), reinterpret_tensor(buf952, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf953, (1024, ), (1, ), 0), buf947, buf948, reinterpret_tensor(buf943, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf944, (4096, ), (1, ), 0), reinterpret_tensor(buf939, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf940, (1024, ), (1, ), 0), buf934, buf935, reinterpret_tensor(buf930, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf931, (1024, ), (1, ), 0), reinterpret_tensor(buf926, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf927, (1024, ), (1, ), 0), reinterpret_tensor(buf922, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf923, (1024, ), (1, ), 0), reinterpret_tensor(buf912, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf913, (1024, ), (1, ), 0), buf907, buf908, reinterpret_tensor(buf903, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf904, (4096, ), (1, ), 0), reinterpret_tensor(buf899, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf900, (1024, ), (1, ), 0), buf894, buf895, reinterpret_tensor(buf890, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf891, (1024, ), (1, ), 0), reinterpret_tensor(buf886, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf887, (1024, ), (1, ), 0), reinterpret_tensor(buf882, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf883, (1024, ), (1, ), 0), reinterpret_tensor(buf872, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf873, (1024, ), (1, ), 0), buf867, buf868, reinterpret_tensor(buf863, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf864, (4096, ), (1, ), 0), reinterpret_tensor(buf859, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf860, (1024, ), (1, ), 0), buf854, buf855, reinterpret_tensor(buf850, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf851, (1024, ), (1, ), 0), reinterpret_tensor(buf846, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf847, (1024, ), (1, ), 0), reinterpret_tensor(buf842, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf843, (1024, ), (1, ), 0), reinterpret_tensor(buf832, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf833, (1024, ), (1, ), 0), buf827, buf828, reinterpret_tensor(buf823, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf824, (4096, ), (1, ), 0), reinterpret_tensor(buf819, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf820, (1024, ), (1, ), 0), buf814, buf815, reinterpret_tensor(buf810, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf811, (1024, ), (1, ), 0), reinterpret_tensor(buf806, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf807, (1024, ), (1, ), 0), reinterpret_tensor(buf802, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf803, (1024, ), (1, ), 0), reinterpret_tensor(buf792, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf793, (1024, ), (1, ), 0), buf787, buf788, reinterpret_tensor(buf783, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf784, (4096, ), (1, ), 0), reinterpret_tensor(buf779, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf780, (1024, ), (1, ), 0), buf774, buf775, reinterpret_tensor(buf770, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf771, (1024, ), (1, ), 0), reinterpret_tensor(buf766, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf767, (1024, ), (1, ), 0), reinterpret_tensor(buf762, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf763, (1024, ), (1, ), 0), reinterpret_tensor(buf752, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf753, (1024, ), (1, ), 0), buf747, buf748, reinterpret_tensor(buf743, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf744, (4096, ), (1, ), 0), reinterpret_tensor(buf739, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf740, (1024, ), (1, ), 0), buf734, buf735, reinterpret_tensor(buf730, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf731, (1024, ), (1, ), 0), reinterpret_tensor(buf726, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf727, (1024, ), (1, ), 0), reinterpret_tensor(buf722, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf723, (1024, ), (1, ), 0), reinterpret_tensor(buf712, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf713, (1024, ), (1, ), 0), buf707, buf708, reinterpret_tensor(buf703, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf704, (4096, ), (1, ), 0), reinterpret_tensor(buf699, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf700, (1024, ), (1, ), 0), buf694, buf695, reinterpret_tensor(buf690, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf691, (1024, ), (1, ), 0), reinterpret_tensor(buf686, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf687, (1024, ), (1, ), 0), reinterpret_tensor(buf682, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf683, (1024, ), (1, ), 0), reinterpret_tensor(buf672, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf673, (1024, ), (1, ), 0), buf667, buf668, reinterpret_tensor(buf663, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf664, (4096, ), (1, ), 0), reinterpret_tensor(buf659, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf660, (1024, ), (1, ), 0), buf654, buf655, reinterpret_tensor(buf650, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf651, (1024, ), (1, ), 0), reinterpret_tensor(buf646, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf647, (1024, ), (1, ), 0), reinterpret_tensor(buf642, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf643, (1024, ), (1, ), 0), reinterpret_tensor(buf632, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf633, (1024, ), (1, ), 0), buf627, buf628, reinterpret_tensor(buf623, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf624, (4096, ), (1, ), 0), reinterpret_tensor(buf619, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf620, (1024, ), (1, ), 0), buf614, buf615, reinterpret_tensor(buf610, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf611, (1024, ), (1, ), 0), reinterpret_tensor(buf606, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf607, (1024, ), (1, ), 0), reinterpret_tensor(buf602, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf603, (1024, ), (1, ), 0), reinterpret_tensor(buf592, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf593, (1024, ), (1, ), 0), buf587, buf588, reinterpret_tensor(buf583, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf584, (4096, ), (1, ), 0), reinterpret_tensor(buf579, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf580, (1024, ), (1, ), 0), buf574, buf575, reinterpret_tensor(buf570, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf571, (1024, ), (1, ), 0), reinterpret_tensor(buf566, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf567, (1024, ), (1, ), 0), reinterpret_tensor(buf562, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf563, (1024, ), (1, ), 0), reinterpret_tensor(buf552, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf553, (1024, ), (1, ), 0), buf547, buf548, reinterpret_tensor(buf543, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf544, (4096, ), (1, ), 0), reinterpret_tensor(buf539, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf540, (1024, ), (1, ), 0), buf534, buf535, reinterpret_tensor(buf530, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf531, (1024, ), (1, ), 0), reinterpret_tensor(buf526, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf527, (1024, ), (1, ), 0), reinterpret_tensor(buf522, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf523, (1024, ), (1, ), 0), reinterpret_tensor(buf512, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf513, (1024, ), (1, ), 0), buf507, buf508, reinterpret_tensor(buf503, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf504, (4096, ), (1, ), 0), reinterpret_tensor(buf499, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf500, (1024, ), (1, ), 0), buf494, buf495, reinterpret_tensor(buf490, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf491, (1024, ), (1, ), 0), reinterpret_tensor(buf486, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf487, (1024, ), (1, ), 0), reinterpret_tensor(buf482, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf483, (1024, ), (1, ), 0), reinterpret_tensor(buf472, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf473, (1024, ), (1, ), 0), buf467, buf468, reinterpret_tensor(buf463, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf464, (4096, ), (1, ), 0), reinterpret_tensor(buf459, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf460, (1024, ), (1, ), 0), buf454, buf455, reinterpret_tensor(buf450, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf451, (1024, ), (1, ), 0), reinterpret_tensor(buf446, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf447, (1024, ), (1, ), 0), reinterpret_tensor(buf442, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf443, (1024, ), (1, ), 0), reinterpret_tensor(buf432, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf433, (1024, ), (1, ), 0), buf427, buf428, reinterpret_tensor(buf423, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf424, (4096, ), (1, ), 0), reinterpret_tensor(buf419, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf420, (1024, ), (1, ), 0), buf414, buf415, reinterpret_tensor(buf410, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf411, (1024, ), (1, ), 0), reinterpret_tensor(buf406, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf407, (1024, ), (1, ), 0), reinterpret_tensor(buf402, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf403, (1024, ), (1, ), 0), reinterpret_tensor(buf392, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf393, (1024, ), (1, ), 0), buf387, buf388, reinterpret_tensor(buf383, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf384, (4096, ), (1, ), 0), reinterpret_tensor(buf379, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf380, (1024, ), (1, ), 0), buf374, buf375, reinterpret_tensor(buf370, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf371, (1024, ), (1, ), 0), reinterpret_tensor(buf366, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf367, (1024, ), (1, ), 0), reinterpret_tensor(buf362, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf363, (1024, ), (1, ), 0), reinterpret_tensor(buf352, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf353, (1024, ), (1, ), 0), buf347, buf348, reinterpret_tensor(buf343, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf344, (4096, ), (1, ), 0), reinterpret_tensor(buf339, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf340, (1024, ), (1, ), 0), buf334, buf335, reinterpret_tensor(buf330, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf331, (1024, ), (1, ), 0), reinterpret_tensor(buf326, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf327, (1024, ), (1, ), 0), reinterpret_tensor(buf322, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf323, (1024, ), (1, ), 0), reinterpret_tensor(buf312, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf313, (1024, ), (1, ), 0), buf307, buf308, reinterpret_tensor(buf303, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf304, (4096, ), (1, ), 0), reinterpret_tensor(buf299, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf300, (1024, ), (1, ), 0), buf294, buf295, reinterpret_tensor(buf290, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf291, (1024, ), (1, ), 0), reinterpret_tensor(buf286, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf287, (1024, ), (1, ), 0), reinterpret_tensor(buf282, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf283, (1024, ), (1, ), 0), reinterpret_tensor(buf272, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf273, (1024, ), (1, ), 0), buf267, buf268, reinterpret_tensor(buf263, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf264, (4096, ), (1, ), 0), reinterpret_tensor(buf259, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf260, (1024, ), (1, ), 0), buf254, buf255, reinterpret_tensor(buf250, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf251, (1024, ), (1, ), 0), reinterpret_tensor(buf246, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf247, (1024, ), (1, ), 0), reinterpret_tensor(buf242, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf243, (1024, ), (1, ), 0), reinterpret_tensor(buf232, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf233, (1024, ), (1, ), 0), buf227, buf228, reinterpret_tensor(buf223, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf224, (4096, ), (1, ), 0), reinterpret_tensor(buf219, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf220, (1024, ), (1, ), 0), buf214, buf215, reinterpret_tensor(buf210, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf211, (1024, ), (1, ), 0), reinterpret_tensor(buf206, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf207, (1024, ), (1, ), 0), reinterpret_tensor(buf202, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf203, (1024, ), (1, ), 0), reinterpret_tensor(buf192, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf193, (1024, ), (1, ), 0), buf187, buf188, reinterpret_tensor(buf183, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf184, (4096, ), (1, ), 0), reinterpret_tensor(buf179, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf180, (1024, ), (1, ), 0), buf174, buf175, reinterpret_tensor(buf170, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf171, (1024, ), (1, ), 0), reinterpret_tensor(buf166, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf167, (1024, ), (1, ), 0), reinterpret_tensor(buf162, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf163, (1024, ), (1, ), 0), reinterpret_tensor(buf152, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf153, (1024, ), (1, ), 0), buf147, buf148, reinterpret_tensor(buf143, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf144, (4096, ), (1, ), 0), reinterpret_tensor(buf139, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf140, (1024, ), (1, ), 0), buf134, buf135, reinterpret_tensor(buf130, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf131, (1024, ), (1, ), 0), reinterpret_tensor(buf126, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf127, (1024, ), (1, ), 0), reinterpret_tensor(buf122, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf123, (1024, ), (1, ), 0), reinterpret_tensor(buf112, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf113, (1024, ), (1, ), 0), buf107, buf108, reinterpret_tensor(buf103, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf104, (4096, ), (1, ), 0), reinterpret_tensor(buf99, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf100, (1024, ), (1, ), 0), buf94, buf95, reinterpret_tensor(buf90, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf91, (1024, ), (1, ), 0), reinterpret_tensor(buf86, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf87, (1024, ), (1, ), 0), reinterpret_tensor(buf82, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf83, (1024, ), (1, ), 0), reinterpret_tensor(buf72, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf73, (1024, ), (1, ), 0), buf67, buf68, reinterpret_tensor(buf63, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf64, (4096, ), (1, ), 0), reinterpret_tensor(buf59, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf60, (1024, ), (1, ), 0), buf54, buf55, reinterpret_tensor(buf50, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf51, (1024, ), (1, ), 0), reinterpret_tensor(buf46, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf47, (1024, ), (1, ), 0), reinterpret_tensor(buf42, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf43, (1024, ), (1, ), 0), reinterpret_tensor(buf32, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf33, (1024, ), (1, ), 0), buf27, buf28, reinterpret_tensor(buf23, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf24, (4096, ), (1, ), 0), reinterpret_tensor(buf19, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf20, (1024, ), (1, ), 0), buf15, buf16, reinterpret_tensor(buf10, (2, 1024), (1024, 1), 0), reinterpret_tensor(buf11, (2, ), (1, ), 0), None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_318 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_324 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_334 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_340 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_350 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_356 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_366 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_372 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_382 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_388 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_393 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    full_default = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    slice_3 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    getitem_1 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_1 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_293 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_139 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_140 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_47 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_141 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_142 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_16 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_3 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_18 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_4 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_20 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_11 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_8 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_22 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_291 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_133 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_134 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_45 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_135 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_136 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_38 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_17 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_10 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_40 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_42 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_21 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_15 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_44 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_289 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_127 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_128 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_43 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_129 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_130 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_60 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_27 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_17 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_62 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_16 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_64 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_31 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_22 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_66 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_287 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_121 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_122 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_41 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_123 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_124 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_82 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_37 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_24 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_84 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_86 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_41 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_29 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_88 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_285 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_115 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_116 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_39 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_117 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_118 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_104 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_47 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_31 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_106 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_28 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_108 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_51 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_36 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_110 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_283 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_109 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_110 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_37 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_111 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_112 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_126 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_57 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_38 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_128 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_130 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_61 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_43 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_132 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_281 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_103 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_104 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_35 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_105 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_106 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_148 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_67 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_45 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_150 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_40 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_152 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_71 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_50 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_154 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_279 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_97 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_98 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_33 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_99 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_100 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_170 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_77 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_52 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_172 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_46 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_174 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_81 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_57 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_176 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_277 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_91 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_92 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_31 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_93 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_94 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_192 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_87 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_59 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_194 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_52 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_196 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_91 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_64 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_198 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_275 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_85 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_86 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_29 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_87 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_88 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_214 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_97 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_66 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_216 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_58 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_218 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_101 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_71 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_220 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_273 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_79 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_80 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_27 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_81 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_82 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_236 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_107 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_73 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_238 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_64 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_240 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_111 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_78 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_242 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_271 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_73 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_74 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_25 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_75 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_76 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_258 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_117 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_80 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_260 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_70 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_262 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_121 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_85 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_264 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_269 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_67 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_68 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_23 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_69 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_70 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_280 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_127 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_87 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_282 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_76 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_284 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_131 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_92 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_286 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_267 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_61 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_62 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_21 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_63 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_64 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_302 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_137 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_94 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_304 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_82 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_306 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_141 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_99 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_308 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_265 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_55 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_56 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_19 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_57 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_58 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_324 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_147 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_101 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_326 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_88 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_328 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_151 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_106 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_330 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_263 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_49 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_50 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_17 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_51 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_52 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_346 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_157 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_108 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_348 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_94 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_350 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_161 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_113 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_352 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_261 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_43 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_44 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_15 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_45 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_46 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_368 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_167 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_115 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_370 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_100 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_372 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_171 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_120 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_374 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_259 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_37 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_38 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_13 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_39 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_40 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_390 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_177 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_122 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_392 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_106 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_394 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_181 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_127 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_396 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_257 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_31 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_32 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_11 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_33 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_34 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_412 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_187 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_129 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_414 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_112 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_416 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_191 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_134 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_418 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_255 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_25 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_26 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_9 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_27 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_28 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_434 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_197 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_136 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_436 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_118 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_438 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_201 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_141 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_440 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_253 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_19 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_20 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_7 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_21 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_22 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_456 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_207 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_143 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_458 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_124 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_460 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_211 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_148 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_462 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_251 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_13 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_14 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_5 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_15 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_16 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_478 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_217 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_150 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_480 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_130 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_482 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_221 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_155 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_484 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_249 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_7 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_8 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_3 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_9 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_10 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_500 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_227 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_157 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_502 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_136 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_504 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_231 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_162 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_506 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_247 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_1 = rand_strided((16, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_2 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_1 = rand_strided((1, 16, 512, 512), (4194304, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_3 = rand_strided((16, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_4 = rand_strided((16, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_522 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_237 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_164 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_524 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    addmm_142 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_526 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    getitem_241 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.bool)
    mul_169 = rand_strided((1, 512, 1024), (524288, 1024, 1), device='cpu', dtype=torch.float32)
    view_528 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    sub_75 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    ne = rand_strided((1, ), (1, ), device='cpu', dtype=torch.bool)
    sub_77 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    ne_3 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.bool)
    ne_6 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.bool)
    where_4 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.int64)
    ne_8 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.bool)
    where_6 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.int64)
    permute_265 = rand_strided((2, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_54 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_269 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_273 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_55 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_277 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_289 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_294 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_298 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_57 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_302 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_306 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_58 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_310 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_322 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_327 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_331 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_60 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_335 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_339 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_61 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_343 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_355 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_360 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_364 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_63 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_368 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_372 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_64 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_376 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_388 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_393 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_397 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_66 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_401 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_405 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_67 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_409 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_421 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_426 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_430 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_69 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_434 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_438 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_70 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_442 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_454 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_459 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_463 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_72 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_467 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_471 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_73 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_475 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_487 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_492 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_496 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_75 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_500 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_504 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_76 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_508 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_520 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_525 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_529 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_78 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_533 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_537 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_79 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_541 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_553 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_558 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_562 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_81 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_566 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_570 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_82 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_574 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_586 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_591 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_595 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_84 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_599 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_603 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_85 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_607 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_619 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_624 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_628 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_87 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_632 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_636 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_88 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_640 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_652 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_657 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_661 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_90 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_665 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_669 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_91 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_673 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_685 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_690 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_694 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_93 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_698 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_702 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_94 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_706 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_718 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_723 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_727 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_96 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_731 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_735 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_97 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_739 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_751 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_756 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_760 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_99 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_764 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_768 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_100 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_772 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_784 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_789 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_793 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_102 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_797 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_801 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_103 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_805 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_817 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_822 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_826 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_105 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_830 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_834 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_106 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_838 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_850 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_855 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_859 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_108 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_863 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_867 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_109 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_871 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_883 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_888 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_892 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_111 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_896 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_900 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_112 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_904 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_916 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_921 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_925 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_114 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_929 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_933 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_115 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_937 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_949 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_954 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_958 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_117 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_962 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_966 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_118 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_970 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_982 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_987 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_991 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_120 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_995 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_999 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_121 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_1003 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_1015 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_1020 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_1024 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_123 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_1028 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_1032 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_124 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_1036 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_1048 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_1053 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_1057 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    div_126 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    tangents_3 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_206, primals_212, primals_222, primals_228, primals_238, primals_244, primals_254, primals_260, primals_270, primals_276, primals_286, primals_292, primals_302, primals_308, primals_318, primals_324, primals_334, primals_340, primals_350, primals_356, primals_366, primals_372, primals_382, primals_388, primals_393, full_default, slice_3, getitem_1, mul_1, view, getitem_293, permute_default_139, permute_default_140, alias_default_47, permute_default_141, permute_default_142, view_16, getitem_7, mul_3, view_18, addmm_4, view_20, getitem_11, mul_8, view_22, getitem_291, permute_default_133, permute_default_134, alias_default_45, permute_default_135, permute_default_136, view_38, getitem_17, mul_10, view_40, addmm_10, view_42, getitem_21, mul_15, view_44, getitem_289, permute_default_127, permute_default_128, alias_default_43, permute_default_129, permute_default_130, view_60, getitem_27, mul_17, view_62, addmm_16, view_64, getitem_31, mul_22, view_66, getitem_287, permute_default_121, permute_default_122, alias_default_41, permute_default_123, permute_default_124, view_82, getitem_37, mul_24, view_84, addmm_22, view_86, getitem_41, mul_29, view_88, getitem_285, permute_default_115, permute_default_116, alias_default_39, permute_default_117, permute_default_118, view_104, getitem_47, mul_31, view_106, addmm_28, view_108, getitem_51, mul_36, view_110, getitem_283, permute_default_109, permute_default_110, alias_default_37, permute_default_111, permute_default_112, view_126, getitem_57, mul_38, view_128, addmm_34, view_130, getitem_61, mul_43, view_132, getitem_281, permute_default_103, permute_default_104, alias_default_35, permute_default_105, permute_default_106, view_148, getitem_67, mul_45, view_150, addmm_40, view_152, getitem_71, mul_50, view_154, getitem_279, permute_default_97, permute_default_98, alias_default_33, permute_default_99, permute_default_100, view_170, getitem_77, mul_52, view_172, addmm_46, view_174, getitem_81, mul_57, view_176, getitem_277, permute_default_91, permute_default_92, alias_default_31, permute_default_93, permute_default_94, view_192, getitem_87, mul_59, view_194, addmm_52, view_196, getitem_91, mul_64, view_198, getitem_275, permute_default_85, permute_default_86, alias_default_29, permute_default_87, permute_default_88, view_214, getitem_97, mul_66, view_216, addmm_58, view_218, getitem_101, mul_71, view_220, getitem_273, permute_default_79, permute_default_80, alias_default_27, permute_default_81, permute_default_82, view_236, getitem_107, mul_73, view_238, addmm_64, view_240, getitem_111, mul_78, view_242, getitem_271, permute_default_73, permute_default_74, alias_default_25, permute_default_75, permute_default_76, view_258, getitem_117, mul_80, view_260, addmm_70, view_262, getitem_121, mul_85, view_264, getitem_269, permute_default_67, permute_default_68, alias_default_23, permute_default_69, permute_default_70, view_280, getitem_127, mul_87, view_282, addmm_76, view_284, getitem_131, mul_92, view_286, getitem_267, permute_default_61, permute_default_62, alias_default_21, permute_default_63, permute_default_64, view_302, getitem_137, mul_94, view_304, addmm_82, view_306, getitem_141, mul_99, view_308, getitem_265, permute_default_55, permute_default_56, alias_default_19, permute_default_57, permute_default_58, view_324, getitem_147, mul_101, view_326, addmm_88, view_328, getitem_151, mul_106, view_330, getitem_263, permute_default_49, permute_default_50, alias_default_17, permute_default_51, permute_default_52, view_346, getitem_157, mul_108, view_348, addmm_94, view_350, getitem_161, mul_113, view_352, getitem_261, permute_default_43, permute_default_44, alias_default_15, permute_default_45, permute_default_46, view_368, getitem_167, mul_115, view_370, addmm_100, view_372, getitem_171, mul_120, view_374, getitem_259, permute_default_37, permute_default_38, alias_default_13, permute_default_39, permute_default_40, view_390, getitem_177, mul_122, view_392, addmm_106, view_394, getitem_181, mul_127, view_396, getitem_257, permute_default_31, permute_default_32, alias_default_11, permute_default_33, permute_default_34, view_412, getitem_187, mul_129, view_414, addmm_112, view_416, getitem_191, mul_134, view_418, getitem_255, permute_default_25, permute_default_26, alias_default_9, permute_default_27, permute_default_28, view_434, getitem_197, mul_136, view_436, addmm_118, view_438, getitem_201, mul_141, view_440, getitem_253, permute_default_19, permute_default_20, alias_default_7, permute_default_21, permute_default_22, view_456, getitem_207, mul_143, view_458, addmm_124, view_460, getitem_211, mul_148, view_462, getitem_251, permute_default_13, permute_default_14, alias_default_5, permute_default_15, permute_default_16, view_478, getitem_217, mul_150, view_480, addmm_130, view_482, getitem_221, mul_155, view_484, getitem_249, permute_default_7, permute_default_8, alias_default_3, permute_default_9, permute_default_10, view_500, getitem_227, mul_157, view_502, addmm_136, view_504, getitem_231, mul_162, view_506, getitem_247, permute_default_1, permute_default_2, alias_default_1, permute_default_3, permute_default_4, view_522, getitem_237, mul_164, view_524, addmm_142, view_526, getitem_241, mul_169, view_528, sub_75, ne, sub_77, ne_3, ne_6, where_4, ne_8, where_6, permute_265, div_54, permute_269, permute_273, div_55, permute_277, permute_289, permute_294, permute_298, div_57, permute_302, permute_306, div_58, permute_310, permute_322, permute_327, permute_331, div_60, permute_335, permute_339, div_61, permute_343, permute_355, permute_360, permute_364, div_63, permute_368, permute_372, div_64, permute_376, permute_388, permute_393, permute_397, div_66, permute_401, permute_405, div_67, permute_409, permute_421, permute_426, permute_430, div_69, permute_434, permute_438, div_70, permute_442, permute_454, permute_459, permute_463, div_72, permute_467, permute_471, div_73, permute_475, permute_487, permute_492, permute_496, div_75, permute_500, permute_504, div_76, permute_508, permute_520, permute_525, permute_529, div_78, permute_533, permute_537, div_79, permute_541, permute_553, permute_558, permute_562, div_81, permute_566, permute_570, div_82, permute_574, permute_586, permute_591, permute_595, div_84, permute_599, permute_603, div_85, permute_607, permute_619, permute_624, permute_628, div_87, permute_632, permute_636, div_88, permute_640, permute_652, permute_657, permute_661, div_90, permute_665, permute_669, div_91, permute_673, permute_685, permute_690, permute_694, div_93, permute_698, permute_702, div_94, permute_706, permute_718, permute_723, permute_727, div_96, permute_731, permute_735, div_97, permute_739, permute_751, permute_756, permute_760, div_99, permute_764, permute_768, div_100, permute_772, permute_784, permute_789, permute_793, div_102, permute_797, permute_801, div_103, permute_805, permute_817, permute_822, permute_826, div_105, permute_830, permute_834, div_106, permute_838, permute_850, permute_855, permute_859, div_108, permute_863, permute_867, div_109, permute_871, permute_883, permute_888, permute_892, div_111, permute_896, permute_900, div_112, permute_904, permute_916, permute_921, permute_925, div_114, permute_929, permute_933, div_115, permute_937, permute_949, permute_954, permute_958, div_117, permute_962, permute_966, div_118, permute_970, permute_982, permute_987, permute_991, div_120, permute_995, permute_999, div_121, permute_1003, permute_1015, permute_1020, permute_1024, div_123, permute_1028, permute_1032, div_124, permute_1036, permute_1048, permute_1053, permute_1057, div_126, tangents_1, tangents_2, tangents_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MegatronBertForQuestionAnswering', benchmark_compiled_module)
