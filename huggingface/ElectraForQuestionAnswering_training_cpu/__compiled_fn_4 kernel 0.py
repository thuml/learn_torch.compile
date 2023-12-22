
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp11 = out_ptr2[static_cast<long>(x0)];
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
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(256.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_6 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(256.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(256.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_14 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(256.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(256.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_22 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(256.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(256.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_30 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(256.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(256.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_38 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(256.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(256.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_46 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(256.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(256.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_54 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(256.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(256.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_62 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(256.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(256.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_70 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(256.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(256.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_78 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(256.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(256.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_86 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(256.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(256.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (256L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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


cpp_fused_sum_94 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_sum_99 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_sum_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const long* in_ptr6,
                       const long* in_ptr7,
                       const long* in_ptr8,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                            auto tmp1 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                            auto tmp6 = in_ptr3[static_cast<long>(x1)];
                            auto tmp8 = in_ptr4[static_cast<long>(x1 + (128L*x0))];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = static_cast<float>(1.1111111111111112);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                            tmp_acc0 = tmp_acc0 + tmp7;
                            tmp_acc1 = tmp_acc1 + tmp9;
                        }
                        out_ptr1[static_cast<long>(x0)] = tmp_acc0;
                        out_ptr2[static_cast<long>(x0)] = tmp_acc1;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr5[static_cast<long>(x0)];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                        auto tmp7 = in_ptr3[static_cast<long>(x1)];
                        auto tmp11 = out_ptr1[static_cast<long>(x0)];
                        auto tmp13 = in_ptr4[static_cast<long>(x1 + (128L*x0))];
                        auto tmp14 = out_ptr2[static_cast<long>(x0)];
                        auto tmp18 = in_ptr6[static_cast<long>(x0)];
                        auto tmp23 = in_ptr7[static_cast<long>(x0)];
                        auto tmp26 = in_ptr8[static_cast<long>(x0)];
                        auto tmp3 = c10::convert<float>(tmp2);
                        auto tmp4 = static_cast<float>(1.1111111111111112);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                        auto tmp9 = static_cast<float>(128.0);
                        auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                        auto tmp12 = decltype(tmp10)(tmp10 - tmp11);
                        auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                        auto tmp16 = decltype(tmp12)(tmp12 - tmp15);
                        auto tmp17 = decltype(tmp0)(tmp0 * tmp16);
                        auto tmp19 = static_cast<long>(-1);
                        auto tmp20 = tmp18 == tmp19;
                        auto tmp21 = static_cast<float>(0.0);
                        auto tmp22 = tmp20 ? tmp21 : tmp17;
                        auto tmp24 = tmp23 == tmp19;
                        auto tmp25 = tmp24 ? tmp21 : tmp17;
                        auto tmp27 = static_cast<long>(0);
                        auto tmp28 = tmp26 == tmp27;
                        auto tmp29 = tmp28 ? tmp21 : tmp17;
                        out_ptr4[static_cast<long>(x1 + (128L*x0))] = tmp22;
                        out_ptr5[static_cast<long>(x1 + (128L*x0))] = tmp25;
                        out_ptr6[static_cast<long>(x1 + (128L*x0))] = tmp29;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x0 + (128L*x1))];
                            auto tmp1 = in_ptr2[static_cast<long>(x0 + (128L*x1))];
                            auto tmp6 = in_ptr4[static_cast<long>(x0 + (128L*x1))];
                            auto tmp2 = c10::convert<float>(tmp1);
                            auto tmp3 = static_cast<float>(1.1111111111111112);
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            tmp_acc0 = tmp_acc0 + tmp7;
                            tmp_acc1 = tmp_acc1 + tmp5;
                        }
                        out_ptr7[static_cast<long>(x0)] = tmp_acc0;
                        out_ptr8[static_cast<long>(x0)] = tmp_acc1;
                    }
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
                    tmp1.store(out_ptr9 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = static_cast<float>(0.0);
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_embedding_dense_backward_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3906816L); x0+=static_cast<long>(8L))
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
    primals_4, primals_16, primals_22, primals_32, primals_38, primals_48, primals_54, primals_64, primals_70, primals_80, primals_86, primals_96, primals_102, primals_112, primals_118, primals_128, primals_134, primals_144, primals_150, primals_160, primals_166, primals_176, primals_182, primals_192, primals_198, primals_204, expand, slice_4, mul_1, getitem_3, view, view_2, getitem_149, permute_default_67, permute_default_68, alias_default_23, permute_default_69, permute_default_70, view_18, getitem_7, mul_3, view_20, addmm_5, view_22, getitem_11, mul_8, view_24, getitem_147, permute_default_61, permute_default_62, alias_default_21, permute_default_63, permute_default_64, view_40, getitem_17, mul_10, view_42, addmm_11, view_44, getitem_21, mul_15, view_46, getitem_145, permute_default_55, permute_default_56, alias_default_19, permute_default_57, permute_default_58, view_62, getitem_27, mul_17, view_64, addmm_17, view_66, getitem_31, mul_22, view_68, getitem_143, permute_default_49, permute_default_50, alias_default_17, permute_default_51, permute_default_52, view_84, getitem_37, mul_24, view_86, addmm_23, view_88, getitem_41, mul_29, view_90, getitem_141, permute_default_43, permute_default_44, alias_default_15, permute_default_45, permute_default_46, view_106, getitem_47, mul_31, view_108, addmm_29, view_110, getitem_51, mul_36, view_112, getitem_139, permute_default_37, permute_default_38, alias_default_13, permute_default_39, permute_default_40, view_128, getitem_57, mul_38, view_130, addmm_35, view_132, getitem_61, mul_43, view_134, getitem_137, permute_default_31, permute_default_32, alias_default_11, permute_default_33, permute_default_34, view_150, getitem_67, mul_45, view_152, addmm_41, view_154, getitem_71, mul_50, view_156, getitem_135, permute_default_25, permute_default_26, alias_default_9, permute_default_27, permute_default_28, view_172, getitem_77, mul_52, view_174, addmm_47, view_176, getitem_81, mul_57, view_178, getitem_133, permute_default_19, permute_default_20, alias_default_7, permute_default_21, permute_default_22, view_194, getitem_87, mul_59, view_196, addmm_53, view_198, getitem_91, mul_64, view_200, getitem_131, permute_default_13, permute_default_14, alias_default_5, permute_default_15, permute_default_16, view_216, getitem_97, mul_66, view_218, addmm_59, view_220, getitem_101, mul_71, view_222, getitem_129, permute_default_7, permute_default_8, alias_default_3, permute_default_9, permute_default_10, view_238, getitem_107, mul_73, view_240, addmm_65, view_242, getitem_111, mul_78, view_244, getitem_127, permute_default_1, permute_default_2, alias_default_1, permute_default_3, permute_default_4, view_260, getitem_117, mul_80, view_262, addmm_71, view_264, getitem_121, mul_85, view_266, sub_39, ne, sub_41, ne_3, ne_6, where_4, ne_8, where_6, permute_134, div_30, permute_138, permute_142, div_31, permute_146, permute_158, permute_163, permute_167, div_33, permute_171, permute_175, div_34, permute_179, permute_191, permute_196, permute_200, div_36, permute_204, permute_208, div_37, permute_212, permute_224, permute_229, permute_233, div_39, permute_237, permute_241, div_40, permute_245, permute_257, permute_262, permute_266, div_42, permute_270, permute_274, div_43, permute_278, permute_290, permute_295, permute_299, div_45, permute_303, permute_307, div_46, permute_311, permute_323, permute_328, permute_332, div_48, permute_336, permute_340, div_49, permute_344, permute_356, permute_361, permute_365, div_51, permute_369, permute_373, div_52, permute_377, permute_389, permute_394, permute_398, div_54, permute_402, permute_406, div_55, permute_410, permute_422, permute_427, permute_431, div_57, permute_435, permute_439, div_58, permute_443, permute_455, permute_460, permute_464, div_60, permute_468, permute_472, div_61, permute_476, permute_488, permute_493, permute_497, div_63, permute_501, permute_505, div_64, permute_509, permute_521, permute_526, permute_530, permute_534, div_66, tangents_1, tangents_2, tangents_3 = args
    args.clear()
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_38, (256, ), (1, ))
    assert_size_stride(primals_48, (256, ), (1, ))
    assert_size_stride(primals_54, (256, ), (1, ))
    assert_size_stride(primals_64, (256, ), (1, ))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_80, (256, ), (1, ))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_96, (256, ), (1, ))
    assert_size_stride(primals_102, (256, ), (1, ))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_118, (256, ), (1, ))
    assert_size_stride(primals_128, (256, ), (1, ))
    assert_size_stride(primals_134, (256, ), (1, ))
    assert_size_stride(primals_144, (256, ), (1, ))
    assert_size_stride(primals_150, (256, ), (1, ))
    assert_size_stride(primals_160, (256, ), (1, ))
    assert_size_stride(primals_166, (256, ), (1, ))
    assert_size_stride(primals_176, (256, ), (1, ))
    assert_size_stride(primals_182, (256, ), (1, ))
    assert_size_stride(primals_192, (256, ), (1, ))
    assert_size_stride(primals_198, (256, ), (1, ))
    assert_size_stride(primals_204, (1, 512), (512, 1))
    assert_size_stride(expand, (1, 512), (512, 1))
    assert_size_stride(slice_4, (1, 512), (512, 1))
    assert_size_stride(mul_1, (1, 512, 128), (65536, 128, 1))
    assert_size_stride(getitem_3, (1, 512, 128), (65536, 128, 1))
    assert_size_stride(view, (512, 128), (128, 1))
    assert_size_stride(view_2, (512, 256), (256, 1))
    assert_size_stride(getitem_149, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_67, (4, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_68, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_23, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_69, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_70, (4, 512, 64), (32768, 64, 1))
    assert_size_stride(view_18, (512, 256), (256, 1))
    assert_size_stride(getitem_7, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_3, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_20, (512, 256), (256, 1))
    assert_size_stride(addmm_5, (512, 1024), (1024, 1))
    assert_size_stride(view_22, (512, 1024), (1024, 1))
    assert_size_stride(getitem_11, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_8, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_24, (512, 256), (256, 1))
    assert_size_stride(getitem_147, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_61, (4, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_62, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_21, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_63, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_64, (4, 512, 64), (32768, 64, 1))
    assert_size_stride(view_40, (512, 256), (256, 1))
    assert_size_stride(getitem_17, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_10, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_42, (512, 256), (256, 1))
    assert_size_stride(addmm_11, (512, 1024), (1024, 1))
    assert_size_stride(view_44, (512, 1024), (1024, 1))
    assert_size_stride(getitem_21, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_15, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_46, (512, 256), (256, 1))
    assert_size_stride(getitem_145, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_55, (4, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_56, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_19, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_57, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_58, (4, 512, 64), (32768, 64, 1))
    assert_size_stride(view_62, (512, 256), (256, 1))
    assert_size_stride(getitem_27, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_17, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_64, (512, 256), (256, 1))
    assert_size_stride(addmm_17, (512, 1024), (1024, 1))
    assert_size_stride(view_66, (512, 1024), (1024, 1))
    assert_size_stride(getitem_31, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_22, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_68, (512, 256), (256, 1))
    assert_size_stride(getitem_143, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_49, (4, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_50, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_17, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_51, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_52, (4, 512, 64), (32768, 64, 1))
    assert_size_stride(view_84, (512, 256), (256, 1))
    assert_size_stride(getitem_37, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_24, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_86, (512, 256), (256, 1))
    assert_size_stride(addmm_23, (512, 1024), (1024, 1))
    assert_size_stride(view_88, (512, 1024), (1024, 1))
    assert_size_stride(getitem_41, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_29, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_90, (512, 256), (256, 1))
    assert_size_stride(getitem_141, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_43, (4, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_44, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_15, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_45, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_46, (4, 512, 64), (32768, 64, 1))
    assert_size_stride(view_106, (512, 256), (256, 1))
    assert_size_stride(getitem_47, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_31, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_108, (512, 256), (256, 1))
    assert_size_stride(addmm_29, (512, 1024), (1024, 1))
    assert_size_stride(view_110, (512, 1024), (1024, 1))
    assert_size_stride(getitem_51, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_36, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_112, (512, 256), (256, 1))
    assert_size_stride(getitem_139, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_37, (4, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_38, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_13, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_39, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_40, (4, 512, 64), (32768, 64, 1))
    assert_size_stride(view_128, (512, 256), (256, 1))
    assert_size_stride(getitem_57, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_38, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_130, (512, 256), (256, 1))
    assert_size_stride(addmm_35, (512, 1024), (1024, 1))
    assert_size_stride(view_132, (512, 1024), (1024, 1))
    assert_size_stride(getitem_61, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_43, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_134, (512, 256), (256, 1))
    assert_size_stride(getitem_137, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_31, (4, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_32, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_11, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_33, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_34, (4, 512, 64), (32768, 64, 1))
    assert_size_stride(view_150, (512, 256), (256, 1))
    assert_size_stride(getitem_67, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_45, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_152, (512, 256), (256, 1))
    assert_size_stride(addmm_41, (512, 1024), (1024, 1))
    assert_size_stride(view_154, (512, 1024), (1024, 1))
    assert_size_stride(getitem_71, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_50, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_156, (512, 256), (256, 1))
    assert_size_stride(getitem_135, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_25, (4, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_26, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_9, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_27, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_28, (4, 512, 64), (32768, 64, 1))
    assert_size_stride(view_172, (512, 256), (256, 1))
    assert_size_stride(getitem_77, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_52, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_174, (512, 256), (256, 1))
    assert_size_stride(addmm_47, (512, 1024), (1024, 1))
    assert_size_stride(view_176, (512, 1024), (1024, 1))
    assert_size_stride(getitem_81, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_57, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_178, (512, 256), (256, 1))
    assert_size_stride(getitem_133, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_19, (4, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_20, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_7, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_21, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_22, (4, 512, 64), (32768, 64, 1))
    assert_size_stride(view_194, (512, 256), (256, 1))
    assert_size_stride(getitem_87, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_59, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_196, (512, 256), (256, 1))
    assert_size_stride(addmm_53, (512, 1024), (1024, 1))
    assert_size_stride(view_198, (512, 1024), (1024, 1))
    assert_size_stride(getitem_91, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_64, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_200, (512, 256), (256, 1))
    assert_size_stride(getitem_131, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_13, (4, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_14, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_5, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_15, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_16, (4, 512, 64), (32768, 64, 1))
    assert_size_stride(view_216, (512, 256), (256, 1))
    assert_size_stride(getitem_97, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_66, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_218, (512, 256), (256, 1))
    assert_size_stride(addmm_59, (512, 1024), (1024, 1))
    assert_size_stride(view_220, (512, 1024), (1024, 1))
    assert_size_stride(getitem_101, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_71, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_222, (512, 256), (256, 1))
    assert_size_stride(getitem_129, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_7, (4, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_8, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_3, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_9, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_10, (4, 512, 64), (32768, 64, 1))
    assert_size_stride(view_238, (512, 256), (256, 1))
    assert_size_stride(getitem_107, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_73, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_240, (512, 256), (256, 1))
    assert_size_stride(addmm_65, (512, 1024), (1024, 1))
    assert_size_stride(view_242, (512, 1024), (1024, 1))
    assert_size_stride(getitem_111, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_78, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_244, (512, 256), (256, 1))
    assert_size_stride(getitem_127, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_1, (4, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_2, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_1, (1, 4, 512, 512), (1048576, 262144, 512, 1))
    assert_size_stride(permute_default_3, (4, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_4, (4, 512, 64), (32768, 64, 1))
    assert_size_stride(view_260, (512, 256), (256, 1))
    assert_size_stride(getitem_117, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_80, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_262, (512, 256), (256, 1))
    assert_size_stride(addmm_71, (512, 1024), (1024, 1))
    assert_size_stride(view_264, (512, 1024), (1024, 1))
    assert_size_stride(getitem_121, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(mul_85, (1, 512, 256), (131072, 256, 1))
    assert_size_stride(view_266, (512, 256), (256, 1))
    assert_size_stride(sub_39, (1, 512), (512, 1))
    assert_size_stride(ne, (1, ), (1, ))
    assert_size_stride(sub_41, (1, 512), (512, 1))
    assert_size_stride(ne_3, (1, ), (1, ))
    assert_size_stride(ne_6, (1, 1), (1, 1))
    assert_size_stride(where_4, (1, 1), (1, 1))
    assert_size_stride(ne_8, (1, 1), (1, 1))
    assert_size_stride(where_6, (1, 1), (1, 1))
    assert_size_stride(permute_134, (2, 256), (256, 1))
    assert_size_stride(div_30, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_138, (256, 1024), (1024, 1))
    assert_size_stride(permute_142, (1024, 256), (256, 1))
    assert_size_stride(div_31, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_146, (256, 256), (256, 1))
    assert_size_stride(permute_158, (256, 256), (256, 1))
    assert_size_stride(permute_163, (256, 256), (256, 1))
    assert_size_stride(permute_167, (256, 256), (256, 1))
    assert_size_stride(div_33, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_171, (256, 1024), (1024, 1))
    assert_size_stride(permute_175, (1024, 256), (256, 1))
    assert_size_stride(div_34, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_179, (256, 256), (256, 1))
    assert_size_stride(permute_191, (256, 256), (256, 1))
    assert_size_stride(permute_196, (256, 256), (256, 1))
    assert_size_stride(permute_200, (256, 256), (256, 1))
    assert_size_stride(div_36, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_204, (256, 1024), (1024, 1))
    assert_size_stride(permute_208, (1024, 256), (256, 1))
    assert_size_stride(div_37, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_212, (256, 256), (256, 1))
    assert_size_stride(permute_224, (256, 256), (256, 1))
    assert_size_stride(permute_229, (256, 256), (256, 1))
    assert_size_stride(permute_233, (256, 256), (256, 1))
    assert_size_stride(div_39, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_237, (256, 1024), (1024, 1))
    assert_size_stride(permute_241, (1024, 256), (256, 1))
    assert_size_stride(div_40, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_245, (256, 256), (256, 1))
    assert_size_stride(permute_257, (256, 256), (256, 1))
    assert_size_stride(permute_262, (256, 256), (256, 1))
    assert_size_stride(permute_266, (256, 256), (256, 1))
    assert_size_stride(div_42, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_270, (256, 1024), (1024, 1))
    assert_size_stride(permute_274, (1024, 256), (256, 1))
    assert_size_stride(div_43, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_278, (256, 256), (256, 1))
    assert_size_stride(permute_290, (256, 256), (256, 1))
    assert_size_stride(permute_295, (256, 256), (256, 1))
    assert_size_stride(permute_299, (256, 256), (256, 1))
    assert_size_stride(div_45, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_303, (256, 1024), (1024, 1))
    assert_size_stride(permute_307, (1024, 256), (256, 1))
    assert_size_stride(div_46, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_311, (256, 256), (256, 1))
    assert_size_stride(permute_323, (256, 256), (256, 1))
    assert_size_stride(permute_328, (256, 256), (256, 1))
    assert_size_stride(permute_332, (256, 256), (256, 1))
    assert_size_stride(div_48, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_336, (256, 1024), (1024, 1))
    assert_size_stride(permute_340, (1024, 256), (256, 1))
    assert_size_stride(div_49, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_344, (256, 256), (256, 1))
    assert_size_stride(permute_356, (256, 256), (256, 1))
    assert_size_stride(permute_361, (256, 256), (256, 1))
    assert_size_stride(permute_365, (256, 256), (256, 1))
    assert_size_stride(div_51, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_369, (256, 1024), (1024, 1))
    assert_size_stride(permute_373, (1024, 256), (256, 1))
    assert_size_stride(div_52, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_377, (256, 256), (256, 1))
    assert_size_stride(permute_389, (256, 256), (256, 1))
    assert_size_stride(permute_394, (256, 256), (256, 1))
    assert_size_stride(permute_398, (256, 256), (256, 1))
    assert_size_stride(div_54, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_402, (256, 1024), (1024, 1))
    assert_size_stride(permute_406, (1024, 256), (256, 1))
    assert_size_stride(div_55, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_410, (256, 256), (256, 1))
    assert_size_stride(permute_422, (256, 256), (256, 1))
    assert_size_stride(permute_427, (256, 256), (256, 1))
    assert_size_stride(permute_431, (256, 256), (256, 1))
    assert_size_stride(div_57, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_435, (256, 1024), (1024, 1))
    assert_size_stride(permute_439, (1024, 256), (256, 1))
    assert_size_stride(div_58, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_443, (256, 256), (256, 1))
    assert_size_stride(permute_455, (256, 256), (256, 1))
    assert_size_stride(permute_460, (256, 256), (256, 1))
    assert_size_stride(permute_464, (256, 256), (256, 1))
    assert_size_stride(div_60, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_468, (256, 1024), (1024, 1))
    assert_size_stride(permute_472, (1024, 256), (256, 1))
    assert_size_stride(div_61, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_476, (256, 256), (256, 1))
    assert_size_stride(permute_488, (256, 256), (256, 1))
    assert_size_stride(permute_493, (256, 256), (256, 1))
    assert_size_stride(permute_497, (256, 256), (256, 1))
    assert_size_stride(div_63, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_501, (256, 1024), (1024, 1))
    assert_size_stride(permute_505, (1024, 256), (256, 1))
    assert_size_stride(div_64, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_509, (256, 256), (256, 1))
    assert_size_stride(permute_521, (256, 256), (256, 1))
    assert_size_stride(permute_526, (256, 256), (256, 1))
    assert_size_stride(permute_530, (256, 256), (256, 1))
    assert_size_stride(permute_534, (256, 128), (128, 1))
    assert_size_stride(div_66, (1, 512, 1), (512, 1, 1))
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
    cpp_fused__log_softmax_backward_data_cat_div_nll_loss_backward_nll_loss_forward_2(c_void_p(buf0.data_ptr()), c_void_p(ne_6.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(ne_3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(ne_8.data_ptr()), c_void_p(ne.data_ptr()), c_void_p(tangents_2.data_ptr()), c_void_p(sub_39.data_ptr()), c_void_p(tangents_3.data_ptr()), c_void_p(sub_41.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()))
    del buf3
    del buf7
    del ne
    del ne_3
    del ne_6
    del ne_8
    del sub_39
    del sub_41
    del tangents_1
    del tangents_2
    del tangents_3
    buf9 = empty((512, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf8, (512, 2), (2, 1), 0), permute_134, out=buf9)
    del permute_134
    buf10 = reinterpret_tensor(buf4, (2, 256), (256, 1), 0); del buf4  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf8, (2, 512), (1, 2), 0), view_266, out=buf10)
    del view_266
    buf11 = empty((1, 2), device='cpu', dtype=torch.float32)
    buf12 = reinterpret_tensor(buf0, (1, 512, 1), (512, 1, 512), 0); del buf0  # reuse
    buf13 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf14 = empty((1, 512, 256), device='cpu', dtype=torch.float32)
    buf15 = empty((256, ), device='cpu', dtype=torch.float32)
    buf16 = empty((256, ), device='cpu', dtype=torch.float32)
    buf17 = empty((1, 512, 256), device='cpu', dtype=torch.float32)
    cpp_fused_native_dropout_backward_native_layer_norm_backward_sum_3(c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(mul_85.data_ptr()), c_void_p(div_30.data_ptr()), c_void_p(getitem_121.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()))
    del div_30
    del getitem_121
    del mul_85
    del primals_198
    buf18 = empty((512, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf17, (512, 256), (256, 1), 0), permute_138, out=buf18)
    del permute_138
    buf19 = empty((256, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf17, (256, 512), (1, 256), 0), view_264, out=buf19)
    del view_264
    buf20 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf21 = reinterpret_tensor(buf18, (1, 512, 1024), (524288, 1024, 1), 0); del buf18  # reuse
    cpp_fused_gelu_gelu_backward_sum_4(c_void_p(buf21.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(addmm_71.data_ptr()), c_void_p(buf20.data_ptr()))
    del addmm_71
    buf22 = reinterpret_tensor(buf17, (512, 256), (256, 1), 0); del buf17  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf21, (512, 1024), (1024, 1), 0), permute_142, out=buf22)
    del permute_142
    buf23 = empty((1024, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf21, (1024, 512), (1, 1024), 0), view_262, out=buf23)
    del view_262
    buf24 = reinterpret_tensor(buf8, (1, 1024), (1024, 1), 0); del buf8  # reuse
    buf25 = buf13; del buf13  # reuse
    buf26 = buf12; del buf12  # reuse
    buf27 = reinterpret_tensor(buf9, (1, 512, 256), (131072, 256, 1), 0); del buf9  # reuse
    buf28 = empty((256, ), device='cpu', dtype=torch.float32)
    buf29 = empty((256, ), device='cpu', dtype=torch.float32)
    buf30 = empty((1, 512, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_5(c_void_p(buf21.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(mul_80.data_ptr()), c_void_p(div_31.data_ptr()), c_void_p(getitem_117.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()))
    del div_31
    del getitem_117
    del mul_80
    del primals_192
    buf31 = buf22; del buf22  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf30, (512, 256), (256, 1), 0), permute_146, out=buf31)
    del permute_146
    buf32 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf30, (256, 512), (1, 256), 0), view_260, out=buf32)
    del view_260
    buf33 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_6(c_void_p(buf30.data_ptr()), c_void_p(buf33.data_ptr()))
    buf34 = reinterpret_tensor(buf30, (4, 512, 64), (32768, 64, 1), 0); del buf30  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_1, reinterpret_tensor(buf31, (4, 512, 64), (64, 256, 1), 0), out=buf34)
    del permute_default_1
    buf35 = empty((4, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf31, (4, 512, 64), (64, 256, 1), 0), permute_default_2, out=buf35)
    del permute_default_2
    buf36 = empty_strided((1, 4, 512, 1), (2048, 512, 1, 2048), device='cpu', dtype=torch.float32)
    buf37 = reinterpret_tensor(buf35, (1, 4, 512, 512), (1048576, 262144, 512, 1), 0); del buf35  # reuse
    cpp_fused_7(c_void_p(buf37.data_ptr()), c_void_p(getitem_127.data_ptr()), c_void_p(alias_default_1.data_ptr()), c_void_p(buf36.data_ptr()))
    del alias_default_1
    del getitem_127
    buf38 = reinterpret_tensor(buf31, (4, 64, 512), (32768, 512, 1), 0); del buf31  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_3, reinterpret_tensor(buf37, (4, 512, 512), (262144, 512, 1), 0), out=buf38)
    del permute_default_3
    buf39 = reinterpret_tensor(buf14, (4, 512, 64), (32768, 64, 1), 0); del buf14  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf37, (4, 512, 512), (262144, 512, 1), 0), permute_default_4, out=buf39)
    del permute_default_4
    buf40 = empty((512, 256), device='cpu', dtype=torch.float32)
    cpp_fused_view_8(c_void_p(buf34.data_ptr()), c_void_p(buf40.data_ptr()))
    buf41 = reinterpret_tensor(buf34, (512, 256), (256, 1), 0); del buf34  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf40, permute_158, out=buf41)
    del permute_158
    buf42 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf40, (256, 512), (1, 256), 0), view_244, out=buf42)
    buf43 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf44 = reinterpret_tensor(buf38, (512, 256), (1, 512), 0); del buf38  # reuse
    cpp_fused_sum_view_9(c_void_p(buf44.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf43.data_ptr()))
    buf45 = buf40; del buf40  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf44, permute_163, out=buf45)
    del permute_163
    buf46 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf44, (256, 512), (512, 1), 0), view_244, out=buf46)
    buf47 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf48 = empty((512, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_view_10(c_void_p(buf44.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    buf49 = reinterpret_tensor(buf44, (512, 256), (256, 1), 0); del buf44  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf48, permute_167, out=buf49)
    del permute_167
    buf50 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf48, (256, 512), (1, 256), 0), view_244, out=buf50)
    del view_244
    buf51 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf52 = reinterpret_tensor(buf39, (1, 512, 256), (131072, 256, 1), 0); del buf39  # reuse
    buf53 = buf26; del buf26  # reuse
    buf54 = buf25; del buf25  # reuse
    buf55 = buf52; del buf52  # reuse
    buf56 = empty((256, ), device='cpu', dtype=torch.float32)
    buf57 = empty((256, ), device='cpu', dtype=torch.float32)
    buf58 = empty((1, 512, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_11(c_void_p(buf55.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(mul_78.data_ptr()), c_void_p(div_33.data_ptr()), c_void_p(getitem_111.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()))
    del div_33
    del getitem_111
    del mul_78
    del primals_182
    buf59 = reinterpret_tensor(buf21, (512, 1024), (1024, 1), 0); del buf21  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf58, (512, 256), (256, 1), 0), permute_171, out=buf59)
    del permute_171
    buf60 = empty((256, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf58, (256, 512), (1, 256), 0), view_242, out=buf60)
    del view_242
    buf61 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf62 = reinterpret_tensor(buf59, (1, 512, 1024), (524288, 1024, 1), 0); del buf59  # reuse
    cpp_fused_gelu_gelu_backward_sum_12(c_void_p(buf62.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(addmm_65.data_ptr()), c_void_p(buf61.data_ptr()))
    del addmm_65
    buf63 = reinterpret_tensor(buf58, (512, 256), (256, 1), 0); del buf58  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf62, (512, 1024), (1024, 1), 0), permute_175, out=buf63)
    del permute_175
    buf64 = empty((1024, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf62, (1024, 512), (1, 1024), 0), view_240, out=buf64)
    del view_240
    buf65 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf66 = buf54; del buf54  # reuse
    buf67 = buf53; del buf53  # reuse
    buf68 = reinterpret_tensor(buf49, (1, 512, 256), (131072, 256, 1), 0); del buf49  # reuse
    buf69 = empty((256, ), device='cpu', dtype=torch.float32)
    buf70 = empty((256, ), device='cpu', dtype=torch.float32)
    buf71 = reinterpret_tensor(buf48, (1, 512, 256), (131072, 256, 1), 0); del buf48  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_13(c_void_p(buf62.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(mul_73.data_ptr()), c_void_p(div_34.data_ptr()), c_void_p(getitem_107.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()))
    del div_34
    del getitem_107
    del mul_73
    del primals_176
    buf72 = buf63; del buf63  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf71, (512, 256), (256, 1), 0), permute_179, out=buf72)
    del permute_179
    buf73 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf71, (256, 512), (1, 256), 0), view_238, out=buf73)
    del view_238
    buf74 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_14(c_void_p(buf71.data_ptr()), c_void_p(buf74.data_ptr()))
    buf75 = reinterpret_tensor(buf71, (4, 512, 64), (32768, 64, 1), 0); del buf71  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_7, reinterpret_tensor(buf72, (4, 512, 64), (64, 256, 1), 0), out=buf75)
    del permute_default_7
    buf76 = reinterpret_tensor(buf37, (4, 512, 512), (262144, 512, 1), 0); del buf37  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf72, (4, 512, 64), (64, 256, 1), 0), permute_default_8, out=buf76)
    del permute_default_8
    buf77 = buf36; del buf36  # reuse
    buf78 = reinterpret_tensor(buf76, (1, 4, 512, 512), (1048576, 262144, 512, 1), 0); del buf76  # reuse
    cpp_fused_15(c_void_p(buf78.data_ptr()), c_void_p(getitem_129.data_ptr()), c_void_p(alias_default_3.data_ptr()), c_void_p(buf77.data_ptr()))
    del alias_default_3
    del getitem_129
    buf79 = reinterpret_tensor(buf72, (4, 64, 512), (32768, 512, 1), 0); del buf72  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_9, reinterpret_tensor(buf78, (4, 512, 512), (262144, 512, 1), 0), out=buf79)
    del permute_default_9
    buf80 = reinterpret_tensor(buf55, (4, 512, 64), (32768, 64, 1), 0); del buf55  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf78, (4, 512, 512), (262144, 512, 1), 0), permute_default_10, out=buf80)
    del permute_default_10
    buf81 = buf45; del buf45  # reuse
    cpp_fused_view_16(c_void_p(buf75.data_ptr()), c_void_p(buf81.data_ptr()))
    buf82 = reinterpret_tensor(buf75, (512, 256), (256, 1), 0); del buf75  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf81, permute_191, out=buf82)
    del permute_191
    buf83 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf81, (256, 512), (1, 256), 0), view_222, out=buf83)
    buf84 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf85 = reinterpret_tensor(buf79, (512, 256), (1, 512), 0); del buf79  # reuse
    cpp_fused_sum_view_17(c_void_p(buf85.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf84.data_ptr()))
    buf86 = buf81; del buf81  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf85, permute_196, out=buf86)
    del permute_196
    buf87 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf85, (256, 512), (512, 1), 0), view_222, out=buf87)
    buf88 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf89 = buf41; del buf41  # reuse
    cpp_fused_sum_view_18(c_void_p(buf85.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()))
    buf90 = reinterpret_tensor(buf85, (512, 256), (256, 1), 0); del buf85  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf89, permute_200, out=buf90)
    del permute_200
    buf91 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf89, (256, 512), (1, 256), 0), view_222, out=buf91)
    del view_222
    buf92 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf93 = reinterpret_tensor(buf80, (1, 512, 256), (131072, 256, 1), 0); del buf80  # reuse
    buf94 = buf67; del buf67  # reuse
    buf95 = buf66; del buf66  # reuse
    buf96 = buf93; del buf93  # reuse
    buf97 = empty((256, ), device='cpu', dtype=torch.float32)
    buf98 = empty((256, ), device='cpu', dtype=torch.float32)
    buf99 = buf27; del buf27  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_19(c_void_p(buf96.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(mul_71.data_ptr()), c_void_p(div_36.data_ptr()), c_void_p(getitem_101.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()))
    del div_36
    del getitem_101
    del mul_71
    del primals_166
    buf100 = reinterpret_tensor(buf62, (512, 1024), (1024, 1), 0); del buf62  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf99, (512, 256), (256, 1), 0), permute_204, out=buf100)
    del permute_204
    buf101 = empty((256, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf99, (256, 512), (1, 256), 0), view_220, out=buf101)
    del view_220
    buf102 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf103 = reinterpret_tensor(buf100, (1, 512, 1024), (524288, 1024, 1), 0); del buf100  # reuse
    cpp_fused_gelu_gelu_backward_sum_20(c_void_p(buf103.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(addmm_59.data_ptr()), c_void_p(buf102.data_ptr()))
    del addmm_59
    buf104 = reinterpret_tensor(buf99, (512, 256), (256, 1), 0); del buf99  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf103, (512, 1024), (1024, 1), 0), permute_208, out=buf104)
    del permute_208
    buf105 = empty((1024, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf103, (1024, 512), (1, 1024), 0), view_218, out=buf105)
    del view_218
    buf106 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf107 = buf95; del buf95  # reuse
    buf108 = buf94; del buf94  # reuse
    buf109 = reinterpret_tensor(buf90, (1, 512, 256), (131072, 256, 1), 0); del buf90  # reuse
    buf110 = empty((256, ), device='cpu', dtype=torch.float32)
    buf111 = empty((256, ), device='cpu', dtype=torch.float32)
    buf112 = reinterpret_tensor(buf89, (1, 512, 256), (131072, 256, 1), 0); del buf89  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_21(c_void_p(buf103.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(mul_66.data_ptr()), c_void_p(div_37.data_ptr()), c_void_p(getitem_97.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()))
    del div_37
    del getitem_97
    del mul_66
    del primals_160
    buf113 = reinterpret_tensor(buf96, (512, 256), (256, 1), 0); del buf96  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf112, (512, 256), (256, 1), 0), permute_212, out=buf113)
    del permute_212
    buf114 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf112, (256, 512), (1, 256), 0), view_216, out=buf114)
    del view_216
    buf115 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_22(c_void_p(buf112.data_ptr()), c_void_p(buf115.data_ptr()))
    buf116 = reinterpret_tensor(buf112, (4, 512, 64), (32768, 64, 1), 0); del buf112  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_13, reinterpret_tensor(buf113, (4, 512, 64), (64, 256, 1), 0), out=buf116)
    del permute_default_13
    buf117 = reinterpret_tensor(buf78, (4, 512, 512), (262144, 512, 1), 0); del buf78  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf113, (4, 512, 64), (64, 256, 1), 0), permute_default_14, out=buf117)
    del permute_default_14
    buf118 = buf77; del buf77  # reuse
    buf119 = reinterpret_tensor(buf117, (1, 4, 512, 512), (1048576, 262144, 512, 1), 0); del buf117  # reuse
    cpp_fused_23(c_void_p(buf119.data_ptr()), c_void_p(getitem_131.data_ptr()), c_void_p(alias_default_5.data_ptr()), c_void_p(buf118.data_ptr()))
    del alias_default_5
    del getitem_131
    buf120 = reinterpret_tensor(buf113, (4, 64, 512), (32768, 512, 1), 0); del buf113  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_15, reinterpret_tensor(buf119, (4, 512, 512), (262144, 512, 1), 0), out=buf120)
    del permute_default_15
    buf121 = reinterpret_tensor(buf104, (4, 512, 64), (32768, 64, 1), 0); del buf104  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf119, (4, 512, 512), (262144, 512, 1), 0), permute_default_16, out=buf121)
    del permute_default_16
    buf122 = buf86; del buf86  # reuse
    cpp_fused_view_24(c_void_p(buf116.data_ptr()), c_void_p(buf122.data_ptr()))
    buf123 = reinterpret_tensor(buf116, (512, 256), (256, 1), 0); del buf116  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf122, permute_224, out=buf123)
    del permute_224
    buf124 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf122, (256, 512), (1, 256), 0), view_200, out=buf124)
    buf125 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf126 = reinterpret_tensor(buf120, (512, 256), (1, 512), 0); del buf120  # reuse
    cpp_fused_sum_view_25(c_void_p(buf126.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf125.data_ptr()))
    buf127 = buf122; del buf122  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf126, permute_229, out=buf127)
    del permute_229
    buf128 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf126, (256, 512), (512, 1), 0), view_200, out=buf128)
    buf129 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf130 = buf82; del buf82  # reuse
    cpp_fused_sum_view_26(c_void_p(buf126.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()))
    buf131 = reinterpret_tensor(buf126, (512, 256), (256, 1), 0); del buf126  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf130, permute_233, out=buf131)
    del permute_233
    buf132 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf130, (256, 512), (1, 256), 0), view_200, out=buf132)
    del view_200
    buf133 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf134 = reinterpret_tensor(buf121, (1, 512, 256), (131072, 256, 1), 0); del buf121  # reuse
    buf135 = buf108; del buf108  # reuse
    buf136 = buf107; del buf107  # reuse
    buf137 = buf134; del buf134  # reuse
    buf138 = empty((256, ), device='cpu', dtype=torch.float32)
    buf139 = empty((256, ), device='cpu', dtype=torch.float32)
    buf140 = buf68; del buf68  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_27(c_void_p(buf137.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(mul_64.data_ptr()), c_void_p(div_39.data_ptr()), c_void_p(getitem_91.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()))
    del div_39
    del getitem_91
    del mul_64
    del primals_150
    buf141 = reinterpret_tensor(buf103, (512, 1024), (1024, 1), 0); del buf103  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf140, (512, 256), (256, 1), 0), permute_237, out=buf141)
    del permute_237
    buf142 = empty((256, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf140, (256, 512), (1, 256), 0), view_198, out=buf142)
    del view_198
    buf143 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf144 = reinterpret_tensor(buf141, (1, 512, 1024), (524288, 1024, 1), 0); del buf141  # reuse
    cpp_fused_gelu_gelu_backward_sum_28(c_void_p(buf144.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(addmm_53.data_ptr()), c_void_p(buf143.data_ptr()))
    del addmm_53
    buf145 = reinterpret_tensor(buf140, (512, 256), (256, 1), 0); del buf140  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf144, (512, 1024), (1024, 1), 0), permute_241, out=buf145)
    del permute_241
    buf146 = empty((1024, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf144, (1024, 512), (1, 1024), 0), view_196, out=buf146)
    del view_196
    buf147 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf148 = buf136; del buf136  # reuse
    buf149 = buf135; del buf135  # reuse
    buf150 = reinterpret_tensor(buf131, (1, 512, 256), (131072, 256, 1), 0); del buf131  # reuse
    buf151 = empty((256, ), device='cpu', dtype=torch.float32)
    buf152 = empty((256, ), device='cpu', dtype=torch.float32)
    buf153 = reinterpret_tensor(buf130, (1, 512, 256), (131072, 256, 1), 0); del buf130  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_29(c_void_p(buf144.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(primals_144.data_ptr()), c_void_p(mul_59.data_ptr()), c_void_p(div_40.data_ptr()), c_void_p(getitem_87.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()))
    del div_40
    del getitem_87
    del mul_59
    del primals_144
    buf154 = buf145; del buf145  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf153, (512, 256), (256, 1), 0), permute_245, out=buf154)
    del permute_245
    buf155 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf153, (256, 512), (1, 256), 0), view_194, out=buf155)
    del view_194
    buf156 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_30(c_void_p(buf153.data_ptr()), c_void_p(buf156.data_ptr()))
    buf157 = reinterpret_tensor(buf153, (4, 512, 64), (32768, 64, 1), 0); del buf153  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_19, reinterpret_tensor(buf154, (4, 512, 64), (64, 256, 1), 0), out=buf157)
    del permute_default_19
    buf158 = reinterpret_tensor(buf119, (4, 512, 512), (262144, 512, 1), 0); del buf119  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf154, (4, 512, 64), (64, 256, 1), 0), permute_default_20, out=buf158)
    del permute_default_20
    buf159 = buf118; del buf118  # reuse
    buf160 = reinterpret_tensor(buf158, (1, 4, 512, 512), (1048576, 262144, 512, 1), 0); del buf158  # reuse
    cpp_fused_31(c_void_p(buf160.data_ptr()), c_void_p(getitem_133.data_ptr()), c_void_p(alias_default_7.data_ptr()), c_void_p(buf159.data_ptr()))
    del alias_default_7
    del getitem_133
    buf161 = reinterpret_tensor(buf154, (4, 64, 512), (32768, 512, 1), 0); del buf154  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_21, reinterpret_tensor(buf160, (4, 512, 512), (262144, 512, 1), 0), out=buf161)
    del permute_default_21
    buf162 = reinterpret_tensor(buf137, (4, 512, 64), (32768, 64, 1), 0); del buf137  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf160, (4, 512, 512), (262144, 512, 1), 0), permute_default_22, out=buf162)
    del permute_default_22
    buf163 = buf127; del buf127  # reuse
    cpp_fused_view_32(c_void_p(buf157.data_ptr()), c_void_p(buf163.data_ptr()))
    buf164 = reinterpret_tensor(buf157, (512, 256), (256, 1), 0); del buf157  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf163, permute_257, out=buf164)
    del permute_257
    buf165 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf163, (256, 512), (1, 256), 0), view_178, out=buf165)
    buf166 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf167 = reinterpret_tensor(buf161, (512, 256), (1, 512), 0); del buf161  # reuse
    cpp_fused_sum_view_33(c_void_p(buf167.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf166.data_ptr()))
    buf168 = buf163; del buf163  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf167, permute_262, out=buf168)
    del permute_262
    buf169 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf167, (256, 512), (512, 1), 0), view_178, out=buf169)
    buf170 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf171 = buf123; del buf123  # reuse
    cpp_fused_sum_view_34(c_void_p(buf167.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()))
    buf172 = reinterpret_tensor(buf167, (512, 256), (256, 1), 0); del buf167  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf171, permute_266, out=buf172)
    del permute_266
    buf173 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf171, (256, 512), (1, 256), 0), view_178, out=buf173)
    del view_178
    buf174 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf175 = reinterpret_tensor(buf162, (1, 512, 256), (131072, 256, 1), 0); del buf162  # reuse
    buf176 = buf149; del buf149  # reuse
    buf177 = buf148; del buf148  # reuse
    buf178 = buf175; del buf175  # reuse
    buf179 = empty((256, ), device='cpu', dtype=torch.float32)
    buf180 = empty((256, ), device='cpu', dtype=torch.float32)
    buf181 = buf109; del buf109  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_35(c_void_p(buf178.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(mul_57.data_ptr()), c_void_p(div_42.data_ptr()), c_void_p(getitem_81.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()))
    del div_42
    del getitem_81
    del mul_57
    del primals_134
    buf182 = reinterpret_tensor(buf144, (512, 1024), (1024, 1), 0); del buf144  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf181, (512, 256), (256, 1), 0), permute_270, out=buf182)
    del permute_270
    buf183 = empty((256, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf181, (256, 512), (1, 256), 0), view_176, out=buf183)
    del view_176
    buf184 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf185 = reinterpret_tensor(buf182, (1, 512, 1024), (524288, 1024, 1), 0); del buf182  # reuse
    cpp_fused_gelu_gelu_backward_sum_36(c_void_p(buf185.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(addmm_47.data_ptr()), c_void_p(buf184.data_ptr()))
    del addmm_47
    buf186 = reinterpret_tensor(buf181, (512, 256), (256, 1), 0); del buf181  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf185, (512, 1024), (1024, 1), 0), permute_274, out=buf186)
    del permute_274
    buf187 = empty((1024, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf185, (1024, 512), (1, 1024), 0), view_174, out=buf187)
    del view_174
    buf188 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf189 = buf177; del buf177  # reuse
    buf190 = buf176; del buf176  # reuse
    buf191 = reinterpret_tensor(buf172, (1, 512, 256), (131072, 256, 1), 0); del buf172  # reuse
    buf192 = empty((256, ), device='cpu', dtype=torch.float32)
    buf193 = empty((256, ), device='cpu', dtype=torch.float32)
    buf194 = reinterpret_tensor(buf171, (1, 512, 256), (131072, 256, 1), 0); del buf171  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_37(c_void_p(buf185.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(mul_52.data_ptr()), c_void_p(div_43.data_ptr()), c_void_p(getitem_77.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()))
    del div_43
    del getitem_77
    del mul_52
    del primals_128
    buf195 = buf186; del buf186  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf194, (512, 256), (256, 1), 0), permute_278, out=buf195)
    del permute_278
    buf196 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf194, (256, 512), (1, 256), 0), view_172, out=buf196)
    del view_172
    buf197 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_38(c_void_p(buf194.data_ptr()), c_void_p(buf197.data_ptr()))
    buf198 = reinterpret_tensor(buf194, (4, 512, 64), (32768, 64, 1), 0); del buf194  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_25, reinterpret_tensor(buf195, (4, 512, 64), (64, 256, 1), 0), out=buf198)
    del permute_default_25
    buf199 = reinterpret_tensor(buf160, (4, 512, 512), (262144, 512, 1), 0); del buf160  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf195, (4, 512, 64), (64, 256, 1), 0), permute_default_26, out=buf199)
    del permute_default_26
    buf200 = buf159; del buf159  # reuse
    buf201 = reinterpret_tensor(buf199, (1, 4, 512, 512), (1048576, 262144, 512, 1), 0); del buf199  # reuse
    cpp_fused_39(c_void_p(buf201.data_ptr()), c_void_p(getitem_135.data_ptr()), c_void_p(alias_default_9.data_ptr()), c_void_p(buf200.data_ptr()))
    del alias_default_9
    del getitem_135
    buf202 = reinterpret_tensor(buf195, (4, 64, 512), (32768, 512, 1), 0); del buf195  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_27, reinterpret_tensor(buf201, (4, 512, 512), (262144, 512, 1), 0), out=buf202)
    del permute_default_27
    buf203 = reinterpret_tensor(buf178, (4, 512, 64), (32768, 64, 1), 0); del buf178  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf201, (4, 512, 512), (262144, 512, 1), 0), permute_default_28, out=buf203)
    del permute_default_28
    buf204 = buf168; del buf168  # reuse
    cpp_fused_view_40(c_void_p(buf198.data_ptr()), c_void_p(buf204.data_ptr()))
    buf205 = reinterpret_tensor(buf198, (512, 256), (256, 1), 0); del buf198  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf204, permute_290, out=buf205)
    del permute_290
    buf206 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf204, (256, 512), (1, 256), 0), view_156, out=buf206)
    buf207 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf208 = reinterpret_tensor(buf202, (512, 256), (1, 512), 0); del buf202  # reuse
    cpp_fused_sum_view_41(c_void_p(buf208.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf207.data_ptr()))
    buf209 = buf204; del buf204  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf208, permute_295, out=buf209)
    del permute_295
    buf210 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf208, (256, 512), (512, 1), 0), view_156, out=buf210)
    buf211 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf212 = buf164; del buf164  # reuse
    cpp_fused_sum_view_42(c_void_p(buf208.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()))
    buf213 = reinterpret_tensor(buf208, (512, 256), (256, 1), 0); del buf208  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf212, permute_299, out=buf213)
    del permute_299
    buf214 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf212, (256, 512), (1, 256), 0), view_156, out=buf214)
    del view_156
    buf215 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf216 = reinterpret_tensor(buf203, (1, 512, 256), (131072, 256, 1), 0); del buf203  # reuse
    buf217 = buf190; del buf190  # reuse
    buf218 = buf189; del buf189  # reuse
    buf219 = buf216; del buf216  # reuse
    buf220 = empty((256, ), device='cpu', dtype=torch.float32)
    buf221 = empty((256, ), device='cpu', dtype=torch.float32)
    buf222 = buf150; del buf150  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_43(c_void_p(buf219.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(mul_50.data_ptr()), c_void_p(div_45.data_ptr()), c_void_p(getitem_71.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()))
    del div_45
    del getitem_71
    del mul_50
    del primals_118
    buf223 = reinterpret_tensor(buf185, (512, 1024), (1024, 1), 0); del buf185  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf222, (512, 256), (256, 1), 0), permute_303, out=buf223)
    del permute_303
    buf224 = empty((256, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf222, (256, 512), (1, 256), 0), view_154, out=buf224)
    del view_154
    buf225 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf226 = reinterpret_tensor(buf223, (1, 512, 1024), (524288, 1024, 1), 0); del buf223  # reuse
    cpp_fused_gelu_gelu_backward_sum_44(c_void_p(buf226.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(addmm_41.data_ptr()), c_void_p(buf225.data_ptr()))
    del addmm_41
    buf227 = reinterpret_tensor(buf222, (512, 256), (256, 1), 0); del buf222  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf226, (512, 1024), (1024, 1), 0), permute_307, out=buf227)
    del permute_307
    buf228 = empty((1024, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf226, (1024, 512), (1, 1024), 0), view_152, out=buf228)
    del view_152
    buf229 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf230 = buf218; del buf218  # reuse
    buf231 = buf217; del buf217  # reuse
    buf232 = reinterpret_tensor(buf213, (1, 512, 256), (131072, 256, 1), 0); del buf213  # reuse
    buf233 = empty((256, ), device='cpu', dtype=torch.float32)
    buf234 = empty((256, ), device='cpu', dtype=torch.float32)
    buf235 = reinterpret_tensor(buf212, (1, 512, 256), (131072, 256, 1), 0); del buf212  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_45(c_void_p(buf226.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(mul_45.data_ptr()), c_void_p(div_46.data_ptr()), c_void_p(getitem_67.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()))
    del div_46
    del getitem_67
    del mul_45
    del primals_112
    buf236 = buf227; del buf227  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf235, (512, 256), (256, 1), 0), permute_311, out=buf236)
    del permute_311
    buf237 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf235, (256, 512), (1, 256), 0), view_150, out=buf237)
    del view_150
    buf238 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_46(c_void_p(buf235.data_ptr()), c_void_p(buf238.data_ptr()))
    buf239 = reinterpret_tensor(buf235, (4, 512, 64), (32768, 64, 1), 0); del buf235  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_31, reinterpret_tensor(buf236, (4, 512, 64), (64, 256, 1), 0), out=buf239)
    del permute_default_31
    buf240 = reinterpret_tensor(buf201, (4, 512, 512), (262144, 512, 1), 0); del buf201  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf236, (4, 512, 64), (64, 256, 1), 0), permute_default_32, out=buf240)
    del permute_default_32
    buf241 = buf200; del buf200  # reuse
    buf242 = reinterpret_tensor(buf240, (1, 4, 512, 512), (1048576, 262144, 512, 1), 0); del buf240  # reuse
    cpp_fused_47(c_void_p(buf242.data_ptr()), c_void_p(getitem_137.data_ptr()), c_void_p(alias_default_11.data_ptr()), c_void_p(buf241.data_ptr()))
    del alias_default_11
    del getitem_137
    buf243 = reinterpret_tensor(buf236, (4, 64, 512), (32768, 512, 1), 0); del buf236  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_33, reinterpret_tensor(buf242, (4, 512, 512), (262144, 512, 1), 0), out=buf243)
    del permute_default_33
    buf244 = reinterpret_tensor(buf219, (4, 512, 64), (32768, 64, 1), 0); del buf219  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf242, (4, 512, 512), (262144, 512, 1), 0), permute_default_34, out=buf244)
    del permute_default_34
    buf245 = buf209; del buf209  # reuse
    cpp_fused_view_48(c_void_p(buf239.data_ptr()), c_void_p(buf245.data_ptr()))
    buf246 = reinterpret_tensor(buf239, (512, 256), (256, 1), 0); del buf239  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf245, permute_323, out=buf246)
    del permute_323
    buf247 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf245, (256, 512), (1, 256), 0), view_134, out=buf247)
    buf248 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf249 = reinterpret_tensor(buf243, (512, 256), (1, 512), 0); del buf243  # reuse
    cpp_fused_sum_view_49(c_void_p(buf249.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf248.data_ptr()))
    buf250 = buf245; del buf245  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf249, permute_328, out=buf250)
    del permute_328
    buf251 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf249, (256, 512), (512, 1), 0), view_134, out=buf251)
    buf252 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf253 = buf205; del buf205  # reuse
    cpp_fused_sum_view_50(c_void_p(buf249.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()))
    buf254 = reinterpret_tensor(buf249, (512, 256), (256, 1), 0); del buf249  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf253, permute_332, out=buf254)
    del permute_332
    buf255 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf253, (256, 512), (1, 256), 0), view_134, out=buf255)
    del view_134
    buf256 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf257 = reinterpret_tensor(buf244, (1, 512, 256), (131072, 256, 1), 0); del buf244  # reuse
    buf258 = buf231; del buf231  # reuse
    buf259 = buf230; del buf230  # reuse
    buf260 = buf257; del buf257  # reuse
    buf261 = empty((256, ), device='cpu', dtype=torch.float32)
    buf262 = empty((256, ), device='cpu', dtype=torch.float32)
    buf263 = buf191; del buf191  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_51(c_void_p(buf260.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(mul_43.data_ptr()), c_void_p(div_48.data_ptr()), c_void_p(getitem_61.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()))
    del div_48
    del getitem_61
    del mul_43
    del primals_102
    buf264 = reinterpret_tensor(buf226, (512, 1024), (1024, 1), 0); del buf226  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf263, (512, 256), (256, 1), 0), permute_336, out=buf264)
    del permute_336
    buf265 = empty((256, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf263, (256, 512), (1, 256), 0), view_132, out=buf265)
    del view_132
    buf266 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf267 = reinterpret_tensor(buf264, (1, 512, 1024), (524288, 1024, 1), 0); del buf264  # reuse
    cpp_fused_gelu_gelu_backward_sum_52(c_void_p(buf267.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(addmm_35.data_ptr()), c_void_p(buf266.data_ptr()))
    del addmm_35
    buf268 = reinterpret_tensor(buf263, (512, 256), (256, 1), 0); del buf263  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf267, (512, 1024), (1024, 1), 0), permute_340, out=buf268)
    del permute_340
    buf269 = empty((1024, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf267, (1024, 512), (1, 1024), 0), view_130, out=buf269)
    del view_130
    buf270 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf271 = buf259; del buf259  # reuse
    buf272 = buf258; del buf258  # reuse
    buf273 = reinterpret_tensor(buf254, (1, 512, 256), (131072, 256, 1), 0); del buf254  # reuse
    buf274 = empty((256, ), device='cpu', dtype=torch.float32)
    buf275 = empty((256, ), device='cpu', dtype=torch.float32)
    buf276 = reinterpret_tensor(buf253, (1, 512, 256), (131072, 256, 1), 0); del buf253  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_53(c_void_p(buf267.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(mul_38.data_ptr()), c_void_p(div_49.data_ptr()), c_void_p(getitem_57.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()))
    del div_49
    del getitem_57
    del mul_38
    del primals_96
    buf277 = buf268; del buf268  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf276, (512, 256), (256, 1), 0), permute_344, out=buf277)
    del permute_344
    buf278 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf276, (256, 512), (1, 256), 0), view_128, out=buf278)
    del view_128
    buf279 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_54(c_void_p(buf276.data_ptr()), c_void_p(buf279.data_ptr()))
    buf280 = reinterpret_tensor(buf276, (4, 512, 64), (32768, 64, 1), 0); del buf276  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_37, reinterpret_tensor(buf277, (4, 512, 64), (64, 256, 1), 0), out=buf280)
    del permute_default_37
    buf281 = reinterpret_tensor(buf242, (4, 512, 512), (262144, 512, 1), 0); del buf242  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf277, (4, 512, 64), (64, 256, 1), 0), permute_default_38, out=buf281)
    del permute_default_38
    buf282 = buf241; del buf241  # reuse
    buf283 = reinterpret_tensor(buf281, (1, 4, 512, 512), (1048576, 262144, 512, 1), 0); del buf281  # reuse
    cpp_fused_55(c_void_p(buf283.data_ptr()), c_void_p(getitem_139.data_ptr()), c_void_p(alias_default_13.data_ptr()), c_void_p(buf282.data_ptr()))
    del alias_default_13
    del getitem_139
    buf284 = reinterpret_tensor(buf277, (4, 64, 512), (32768, 512, 1), 0); del buf277  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_39, reinterpret_tensor(buf283, (4, 512, 512), (262144, 512, 1), 0), out=buf284)
    del permute_default_39
    buf285 = reinterpret_tensor(buf260, (4, 512, 64), (32768, 64, 1), 0); del buf260  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf283, (4, 512, 512), (262144, 512, 1), 0), permute_default_40, out=buf285)
    del permute_default_40
    buf286 = buf250; del buf250  # reuse
    cpp_fused_view_56(c_void_p(buf280.data_ptr()), c_void_p(buf286.data_ptr()))
    buf287 = reinterpret_tensor(buf280, (512, 256), (256, 1), 0); del buf280  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf286, permute_356, out=buf287)
    del permute_356
    buf288 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf286, (256, 512), (1, 256), 0), view_112, out=buf288)
    buf289 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf290 = reinterpret_tensor(buf284, (512, 256), (1, 512), 0); del buf284  # reuse
    cpp_fused_sum_view_57(c_void_p(buf290.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf289.data_ptr()))
    buf291 = buf286; del buf286  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf290, permute_361, out=buf291)
    del permute_361
    buf292 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf290, (256, 512), (512, 1), 0), view_112, out=buf292)
    buf293 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf294 = buf246; del buf246  # reuse
    cpp_fused_sum_view_58(c_void_p(buf290.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()))
    buf295 = reinterpret_tensor(buf290, (512, 256), (256, 1), 0); del buf290  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf294, permute_365, out=buf295)
    del permute_365
    buf296 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf294, (256, 512), (1, 256), 0), view_112, out=buf296)
    del view_112
    buf297 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf298 = reinterpret_tensor(buf285, (1, 512, 256), (131072, 256, 1), 0); del buf285  # reuse
    buf299 = buf272; del buf272  # reuse
    buf300 = buf271; del buf271  # reuse
    buf301 = buf298; del buf298  # reuse
    buf302 = empty((256, ), device='cpu', dtype=torch.float32)
    buf303 = empty((256, ), device='cpu', dtype=torch.float32)
    buf304 = buf232; del buf232  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_59(c_void_p(buf301.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(mul_36.data_ptr()), c_void_p(div_51.data_ptr()), c_void_p(getitem_51.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()))
    del div_51
    del getitem_51
    del mul_36
    del primals_86
    buf305 = reinterpret_tensor(buf267, (512, 1024), (1024, 1), 0); del buf267  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf304, (512, 256), (256, 1), 0), permute_369, out=buf305)
    del permute_369
    buf306 = empty((256, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf304, (256, 512), (1, 256), 0), view_110, out=buf306)
    del view_110
    buf307 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf308 = reinterpret_tensor(buf305, (1, 512, 1024), (524288, 1024, 1), 0); del buf305  # reuse
    cpp_fused_gelu_gelu_backward_sum_60(c_void_p(buf308.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(addmm_29.data_ptr()), c_void_p(buf307.data_ptr()))
    del addmm_29
    buf309 = reinterpret_tensor(buf304, (512, 256), (256, 1), 0); del buf304  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf308, (512, 1024), (1024, 1), 0), permute_373, out=buf309)
    del permute_373
    buf310 = empty((1024, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf308, (1024, 512), (1, 1024), 0), view_108, out=buf310)
    del view_108
    buf311 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf312 = buf300; del buf300  # reuse
    buf313 = buf299; del buf299  # reuse
    buf314 = reinterpret_tensor(buf295, (1, 512, 256), (131072, 256, 1), 0); del buf295  # reuse
    buf315 = empty((256, ), device='cpu', dtype=torch.float32)
    buf316 = empty((256, ), device='cpu', dtype=torch.float32)
    buf317 = reinterpret_tensor(buf294, (1, 512, 256), (131072, 256, 1), 0); del buf294  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_61(c_void_p(buf308.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(mul_31.data_ptr()), c_void_p(div_52.data_ptr()), c_void_p(getitem_47.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()))
    del div_52
    del getitem_47
    del mul_31
    del primals_80
    buf318 = buf309; del buf309  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf317, (512, 256), (256, 1), 0), permute_377, out=buf318)
    del permute_377
    buf319 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf317, (256, 512), (1, 256), 0), view_106, out=buf319)
    del view_106
    buf320 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_62(c_void_p(buf317.data_ptr()), c_void_p(buf320.data_ptr()))
    buf321 = reinterpret_tensor(buf317, (4, 512, 64), (32768, 64, 1), 0); del buf317  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_43, reinterpret_tensor(buf318, (4, 512, 64), (64, 256, 1), 0), out=buf321)
    del permute_default_43
    buf322 = reinterpret_tensor(buf283, (4, 512, 512), (262144, 512, 1), 0); del buf283  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf318, (4, 512, 64), (64, 256, 1), 0), permute_default_44, out=buf322)
    del permute_default_44
    buf323 = buf282; del buf282  # reuse
    buf324 = reinterpret_tensor(buf322, (1, 4, 512, 512), (1048576, 262144, 512, 1), 0); del buf322  # reuse
    cpp_fused_63(c_void_p(buf324.data_ptr()), c_void_p(getitem_141.data_ptr()), c_void_p(alias_default_15.data_ptr()), c_void_p(buf323.data_ptr()))
    del alias_default_15
    del getitem_141
    buf325 = reinterpret_tensor(buf318, (4, 64, 512), (32768, 512, 1), 0); del buf318  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_45, reinterpret_tensor(buf324, (4, 512, 512), (262144, 512, 1), 0), out=buf325)
    del permute_default_45
    buf326 = reinterpret_tensor(buf301, (4, 512, 64), (32768, 64, 1), 0); del buf301  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf324, (4, 512, 512), (262144, 512, 1), 0), permute_default_46, out=buf326)
    del permute_default_46
    buf327 = buf291; del buf291  # reuse
    cpp_fused_view_64(c_void_p(buf321.data_ptr()), c_void_p(buf327.data_ptr()))
    buf328 = reinterpret_tensor(buf321, (512, 256), (256, 1), 0); del buf321  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf327, permute_389, out=buf328)
    del permute_389
    buf329 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf327, (256, 512), (1, 256), 0), view_90, out=buf329)
    buf330 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf331 = reinterpret_tensor(buf325, (512, 256), (1, 512), 0); del buf325  # reuse
    cpp_fused_sum_view_65(c_void_p(buf331.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf330.data_ptr()))
    buf332 = buf327; del buf327  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf331, permute_394, out=buf332)
    del permute_394
    buf333 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf331, (256, 512), (512, 1), 0), view_90, out=buf333)
    buf334 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf335 = buf287; del buf287  # reuse
    cpp_fused_sum_view_66(c_void_p(buf331.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()))
    buf336 = reinterpret_tensor(buf331, (512, 256), (256, 1), 0); del buf331  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf335, permute_398, out=buf336)
    del permute_398
    buf337 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf335, (256, 512), (1, 256), 0), view_90, out=buf337)
    del view_90
    buf338 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf339 = reinterpret_tensor(buf326, (1, 512, 256), (131072, 256, 1), 0); del buf326  # reuse
    buf340 = buf313; del buf313  # reuse
    buf341 = buf312; del buf312  # reuse
    buf342 = buf339; del buf339  # reuse
    buf343 = empty((256, ), device='cpu', dtype=torch.float32)
    buf344 = empty((256, ), device='cpu', dtype=torch.float32)
    buf345 = buf273; del buf273  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_67(c_void_p(buf342.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(mul_29.data_ptr()), c_void_p(div_54.data_ptr()), c_void_p(getitem_41.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()))
    del div_54
    del getitem_41
    del mul_29
    del primals_70
    buf346 = reinterpret_tensor(buf308, (512, 1024), (1024, 1), 0); del buf308  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf345, (512, 256), (256, 1), 0), permute_402, out=buf346)
    del permute_402
    buf347 = empty((256, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf345, (256, 512), (1, 256), 0), view_88, out=buf347)
    del view_88
    buf348 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf349 = reinterpret_tensor(buf346, (1, 512, 1024), (524288, 1024, 1), 0); del buf346  # reuse
    cpp_fused_gelu_gelu_backward_sum_68(c_void_p(buf349.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(addmm_23.data_ptr()), c_void_p(buf348.data_ptr()))
    del addmm_23
    buf350 = reinterpret_tensor(buf345, (512, 256), (256, 1), 0); del buf345  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf349, (512, 1024), (1024, 1), 0), permute_406, out=buf350)
    del permute_406
    buf351 = empty((1024, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf349, (1024, 512), (1, 1024), 0), view_86, out=buf351)
    del view_86
    buf352 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf353 = buf341; del buf341  # reuse
    buf354 = buf340; del buf340  # reuse
    buf355 = reinterpret_tensor(buf336, (1, 512, 256), (131072, 256, 1), 0); del buf336  # reuse
    buf356 = empty((256, ), device='cpu', dtype=torch.float32)
    buf357 = empty((256, ), device='cpu', dtype=torch.float32)
    buf358 = reinterpret_tensor(buf335, (1, 512, 256), (131072, 256, 1), 0); del buf335  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_69(c_void_p(buf349.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(mul_24.data_ptr()), c_void_p(div_55.data_ptr()), c_void_p(getitem_37.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()))
    del div_55
    del getitem_37
    del mul_24
    del primals_64
    buf359 = buf350; del buf350  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf358, (512, 256), (256, 1), 0), permute_410, out=buf359)
    del permute_410
    buf360 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf358, (256, 512), (1, 256), 0), view_84, out=buf360)
    del view_84
    buf361 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_70(c_void_p(buf358.data_ptr()), c_void_p(buf361.data_ptr()))
    buf362 = reinterpret_tensor(buf358, (4, 512, 64), (32768, 64, 1), 0); del buf358  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_49, reinterpret_tensor(buf359, (4, 512, 64), (64, 256, 1), 0), out=buf362)
    del permute_default_49
    buf363 = reinterpret_tensor(buf324, (4, 512, 512), (262144, 512, 1), 0); del buf324  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf359, (4, 512, 64), (64, 256, 1), 0), permute_default_50, out=buf363)
    del permute_default_50
    buf364 = buf323; del buf323  # reuse
    buf365 = reinterpret_tensor(buf363, (1, 4, 512, 512), (1048576, 262144, 512, 1), 0); del buf363  # reuse
    cpp_fused_71(c_void_p(buf365.data_ptr()), c_void_p(getitem_143.data_ptr()), c_void_p(alias_default_17.data_ptr()), c_void_p(buf364.data_ptr()))
    del alias_default_17
    del getitem_143
    buf366 = reinterpret_tensor(buf359, (4, 64, 512), (32768, 512, 1), 0); del buf359  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_51, reinterpret_tensor(buf365, (4, 512, 512), (262144, 512, 1), 0), out=buf366)
    del permute_default_51
    buf367 = reinterpret_tensor(buf342, (4, 512, 64), (32768, 64, 1), 0); del buf342  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf365, (4, 512, 512), (262144, 512, 1), 0), permute_default_52, out=buf367)
    del permute_default_52
    buf368 = buf332; del buf332  # reuse
    cpp_fused_view_72(c_void_p(buf362.data_ptr()), c_void_p(buf368.data_ptr()))
    buf369 = reinterpret_tensor(buf362, (512, 256), (256, 1), 0); del buf362  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf368, permute_422, out=buf369)
    del permute_422
    buf370 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf368, (256, 512), (1, 256), 0), view_68, out=buf370)
    buf371 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf372 = reinterpret_tensor(buf366, (512, 256), (1, 512), 0); del buf366  # reuse
    cpp_fused_sum_view_73(c_void_p(buf372.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf371.data_ptr()))
    buf373 = buf368; del buf368  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf372, permute_427, out=buf373)
    del permute_427
    buf374 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf372, (256, 512), (512, 1), 0), view_68, out=buf374)
    buf375 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf376 = buf328; del buf328  # reuse
    cpp_fused_sum_view_74(c_void_p(buf372.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()))
    buf377 = reinterpret_tensor(buf372, (512, 256), (256, 1), 0); del buf372  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf376, permute_431, out=buf377)
    del permute_431
    buf378 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf376, (256, 512), (1, 256), 0), view_68, out=buf378)
    del view_68
    buf379 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf380 = reinterpret_tensor(buf367, (1, 512, 256), (131072, 256, 1), 0); del buf367  # reuse
    buf381 = buf354; del buf354  # reuse
    buf382 = buf353; del buf353  # reuse
    buf383 = buf380; del buf380  # reuse
    buf384 = empty((256, ), device='cpu', dtype=torch.float32)
    buf385 = empty((256, ), device='cpu', dtype=torch.float32)
    buf386 = buf314; del buf314  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_75(c_void_p(buf383.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(mul_22.data_ptr()), c_void_p(div_57.data_ptr()), c_void_p(getitem_31.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()))
    del div_57
    del getitem_31
    del mul_22
    del primals_54
    buf387 = reinterpret_tensor(buf349, (512, 1024), (1024, 1), 0); del buf349  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf386, (512, 256), (256, 1), 0), permute_435, out=buf387)
    del permute_435
    buf388 = empty((256, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf386, (256, 512), (1, 256), 0), view_66, out=buf388)
    del view_66
    buf389 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf390 = reinterpret_tensor(buf387, (1, 512, 1024), (524288, 1024, 1), 0); del buf387  # reuse
    cpp_fused_gelu_gelu_backward_sum_76(c_void_p(buf390.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(addmm_17.data_ptr()), c_void_p(buf389.data_ptr()))
    del addmm_17
    buf391 = reinterpret_tensor(buf386, (512, 256), (256, 1), 0); del buf386  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf390, (512, 1024), (1024, 1), 0), permute_439, out=buf391)
    del permute_439
    buf392 = empty((1024, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf390, (1024, 512), (1, 1024), 0), view_64, out=buf392)
    del view_64
    buf393 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf394 = buf382; del buf382  # reuse
    buf395 = buf381; del buf381  # reuse
    buf396 = reinterpret_tensor(buf377, (1, 512, 256), (131072, 256, 1), 0); del buf377  # reuse
    buf397 = empty((256, ), device='cpu', dtype=torch.float32)
    buf398 = empty((256, ), device='cpu', dtype=torch.float32)
    buf399 = reinterpret_tensor(buf376, (1, 512, 256), (131072, 256, 1), 0); del buf376  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_77(c_void_p(buf390.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(mul_17.data_ptr()), c_void_p(div_58.data_ptr()), c_void_p(getitem_27.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()))
    del div_58
    del getitem_27
    del mul_17
    del primals_48
    buf400 = buf391; del buf391  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf399, (512, 256), (256, 1), 0), permute_443, out=buf400)
    del permute_443
    buf401 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf399, (256, 512), (1, 256), 0), view_62, out=buf401)
    del view_62
    buf402 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_78(c_void_p(buf399.data_ptr()), c_void_p(buf402.data_ptr()))
    buf403 = reinterpret_tensor(buf399, (4, 512, 64), (32768, 64, 1), 0); del buf399  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_55, reinterpret_tensor(buf400, (4, 512, 64), (64, 256, 1), 0), out=buf403)
    del permute_default_55
    buf404 = reinterpret_tensor(buf365, (4, 512, 512), (262144, 512, 1), 0); del buf365  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf400, (4, 512, 64), (64, 256, 1), 0), permute_default_56, out=buf404)
    del permute_default_56
    buf405 = buf364; del buf364  # reuse
    buf406 = reinterpret_tensor(buf404, (1, 4, 512, 512), (1048576, 262144, 512, 1), 0); del buf404  # reuse
    cpp_fused_79(c_void_p(buf406.data_ptr()), c_void_p(getitem_145.data_ptr()), c_void_p(alias_default_19.data_ptr()), c_void_p(buf405.data_ptr()))
    del alias_default_19
    del getitem_145
    buf407 = reinterpret_tensor(buf400, (4, 64, 512), (32768, 512, 1), 0); del buf400  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_57, reinterpret_tensor(buf406, (4, 512, 512), (262144, 512, 1), 0), out=buf407)
    del permute_default_57
    buf408 = reinterpret_tensor(buf383, (4, 512, 64), (32768, 64, 1), 0); del buf383  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf406, (4, 512, 512), (262144, 512, 1), 0), permute_default_58, out=buf408)
    del permute_default_58
    buf409 = buf373; del buf373  # reuse
    cpp_fused_view_80(c_void_p(buf403.data_ptr()), c_void_p(buf409.data_ptr()))
    buf410 = reinterpret_tensor(buf403, (512, 256), (256, 1), 0); del buf403  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf409, permute_455, out=buf410)
    del permute_455
    buf411 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf409, (256, 512), (1, 256), 0), view_46, out=buf411)
    buf412 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf413 = reinterpret_tensor(buf407, (512, 256), (1, 512), 0); del buf407  # reuse
    cpp_fused_sum_view_81(c_void_p(buf413.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf412.data_ptr()))
    buf414 = buf409; del buf409  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf413, permute_460, out=buf414)
    del permute_460
    buf415 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf413, (256, 512), (512, 1), 0), view_46, out=buf415)
    buf416 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf417 = buf369; del buf369  # reuse
    cpp_fused_sum_view_82(c_void_p(buf413.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()))
    buf418 = reinterpret_tensor(buf413, (512, 256), (256, 1), 0); del buf413  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf417, permute_464, out=buf418)
    del permute_464
    buf419 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf417, (256, 512), (1, 256), 0), view_46, out=buf419)
    del view_46
    buf420 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf421 = reinterpret_tensor(buf408, (1, 512, 256), (131072, 256, 1), 0); del buf408  # reuse
    buf422 = buf395; del buf395  # reuse
    buf423 = buf394; del buf394  # reuse
    buf424 = buf421; del buf421  # reuse
    buf425 = empty((256, ), device='cpu', dtype=torch.float32)
    buf426 = empty((256, ), device='cpu', dtype=torch.float32)
    buf427 = buf355; del buf355  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_83(c_void_p(buf424.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(mul_15.data_ptr()), c_void_p(div_60.data_ptr()), c_void_p(getitem_21.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()))
    del div_60
    del getitem_21
    del mul_15
    del primals_38
    buf428 = reinterpret_tensor(buf390, (512, 1024), (1024, 1), 0); del buf390  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf427, (512, 256), (256, 1), 0), permute_468, out=buf428)
    del permute_468
    buf429 = empty((256, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf427, (256, 512), (1, 256), 0), view_44, out=buf429)
    del view_44
    buf430 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf431 = reinterpret_tensor(buf428, (1, 512, 1024), (524288, 1024, 1), 0); del buf428  # reuse
    cpp_fused_gelu_gelu_backward_sum_84(c_void_p(buf431.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(addmm_11.data_ptr()), c_void_p(buf430.data_ptr()))
    del addmm_11
    buf432 = reinterpret_tensor(buf427, (512, 256), (256, 1), 0); del buf427  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf431, (512, 1024), (1024, 1), 0), permute_472, out=buf432)
    del permute_472
    buf433 = empty((1024, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf431, (1024, 512), (1, 1024), 0), view_42, out=buf433)
    del view_42
    buf434 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf435 = buf423; del buf423  # reuse
    buf436 = buf422; del buf422  # reuse
    buf437 = reinterpret_tensor(buf418, (1, 512, 256), (131072, 256, 1), 0); del buf418  # reuse
    buf438 = empty((256, ), device='cpu', dtype=torch.float32)
    buf439 = empty((256, ), device='cpu', dtype=torch.float32)
    buf440 = reinterpret_tensor(buf417, (1, 512, 256), (131072, 256, 1), 0); del buf417  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_85(c_void_p(buf431.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(mul_10.data_ptr()), c_void_p(div_61.data_ptr()), c_void_p(getitem_17.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()))
    del div_61
    del getitem_17
    del mul_10
    del primals_32
    buf441 = buf432; del buf432  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf440, (512, 256), (256, 1), 0), permute_476, out=buf441)
    del permute_476
    buf442 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf440, (256, 512), (1, 256), 0), view_40, out=buf442)
    del view_40
    buf443 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_86(c_void_p(buf440.data_ptr()), c_void_p(buf443.data_ptr()))
    buf444 = reinterpret_tensor(buf440, (4, 512, 64), (32768, 64, 1), 0); del buf440  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_61, reinterpret_tensor(buf441, (4, 512, 64), (64, 256, 1), 0), out=buf444)
    del permute_default_61
    buf445 = reinterpret_tensor(buf406, (4, 512, 512), (262144, 512, 1), 0); del buf406  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf441, (4, 512, 64), (64, 256, 1), 0), permute_default_62, out=buf445)
    del permute_default_62
    buf446 = buf405; del buf405  # reuse
    buf447 = reinterpret_tensor(buf445, (1, 4, 512, 512), (1048576, 262144, 512, 1), 0); del buf445  # reuse
    cpp_fused_87(c_void_p(buf447.data_ptr()), c_void_p(getitem_147.data_ptr()), c_void_p(alias_default_21.data_ptr()), c_void_p(buf446.data_ptr()))
    del alias_default_21
    del getitem_147
    buf448 = reinterpret_tensor(buf441, (4, 64, 512), (32768, 512, 1), 0); del buf441  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_63, reinterpret_tensor(buf447, (4, 512, 512), (262144, 512, 1), 0), out=buf448)
    del permute_default_63
    buf449 = reinterpret_tensor(buf424, (4, 512, 64), (32768, 64, 1), 0); del buf424  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf447, (4, 512, 512), (262144, 512, 1), 0), permute_default_64, out=buf449)
    del permute_default_64
    buf450 = buf414; del buf414  # reuse
    cpp_fused_view_88(c_void_p(buf444.data_ptr()), c_void_p(buf450.data_ptr()))
    buf451 = reinterpret_tensor(buf444, (512, 256), (256, 1), 0); del buf444  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf450, permute_488, out=buf451)
    del permute_488
    buf452 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf450, (256, 512), (1, 256), 0), view_24, out=buf452)
    buf453 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf454 = reinterpret_tensor(buf448, (512, 256), (1, 512), 0); del buf448  # reuse
    cpp_fused_sum_view_89(c_void_p(buf454.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf453.data_ptr()))
    buf455 = buf450; del buf450  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf454, permute_493, out=buf455)
    del permute_493
    buf456 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf454, (256, 512), (512, 1), 0), view_24, out=buf456)
    buf457 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf458 = buf410; del buf410  # reuse
    cpp_fused_sum_view_90(c_void_p(buf454.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()))
    buf459 = reinterpret_tensor(buf454, (512, 256), (256, 1), 0); del buf454  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf458, permute_497, out=buf459)
    del permute_497
    buf460 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf458, (256, 512), (1, 256), 0), view_24, out=buf460)
    del view_24
    buf461 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf462 = reinterpret_tensor(buf449, (1, 512, 256), (131072, 256, 1), 0); del buf449  # reuse
    buf463 = buf436; del buf436  # reuse
    buf464 = buf435; del buf435  # reuse
    buf465 = buf462; del buf462  # reuse
    buf466 = empty((256, ), device='cpu', dtype=torch.float32)
    buf467 = empty((256, ), device='cpu', dtype=torch.float32)
    buf468 = buf396; del buf396  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_91(c_void_p(buf465.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_8.data_ptr()), c_void_p(div_63.data_ptr()), c_void_p(getitem_11.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()))
    del buf437
    del div_63
    del getitem_11
    del mul_8
    del primals_22
    buf469 = reinterpret_tensor(buf431, (512, 1024), (1024, 1), 0); del buf431  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf468, (512, 256), (256, 1), 0), permute_501, out=buf469)
    del permute_501
    buf470 = empty((256, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf468, (256, 512), (1, 256), 0), view_22, out=buf470)
    del view_22
    buf471 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf472 = reinterpret_tensor(buf469, (1, 512, 1024), (524288, 1024, 1), 0); del buf469  # reuse
    cpp_fused_gelu_gelu_backward_sum_92(c_void_p(buf472.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(addmm_5.data_ptr()), c_void_p(buf471.data_ptr()))
    del addmm_5
    buf473 = reinterpret_tensor(buf468, (512, 256), (256, 1), 0); del buf468  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf472, (512, 1024), (1024, 1), 0), permute_505, out=buf473)
    del permute_505
    buf474 = empty((1024, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf472, (1024, 512), (1, 1024), 0), view_20, out=buf474)
    del view_20
    buf475 = empty((1, 1024), device='cpu', dtype=torch.float32)
    buf476 = buf464; del buf464  # reuse
    buf477 = buf463; del buf463  # reuse
    buf478 = reinterpret_tensor(buf459, (1, 512, 256), (131072, 256, 1), 0); del buf459  # reuse
    buf479 = empty((256, ), device='cpu', dtype=torch.float32)
    buf480 = empty((256, ), device='cpu', dtype=torch.float32)
    buf481 = reinterpret_tensor(buf458, (1, 512, 256), (131072, 256, 1), 0); del buf458  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_93(c_void_p(buf472.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_3.data_ptr()), c_void_p(div_64.data_ptr()), c_void_p(getitem_7.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf481.data_ptr()))
    del buf472
    del div_64
    del getitem_7
    del mul_3
    del primals_16
    buf482 = buf473; del buf473  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf481, (512, 256), (256, 1), 0), permute_509, out=buf482)
    del permute_509
    buf483 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf481, (256, 512), (1, 256), 0), view_18, out=buf483)
    del view_18
    buf484 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_94(c_void_p(buf481.data_ptr()), c_void_p(buf484.data_ptr()))
    buf485 = reinterpret_tensor(buf481, (4, 512, 64), (32768, 64, 1), 0); del buf481  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_67, reinterpret_tensor(buf482, (4, 512, 64), (64, 256, 1), 0), out=buf485)
    del permute_default_67
    buf486 = reinterpret_tensor(buf447, (4, 512, 512), (262144, 512, 1), 0); del buf447  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf482, (4, 512, 64), (64, 256, 1), 0), permute_default_68, out=buf486)
    del permute_default_68
    buf487 = buf446; del buf446  # reuse
    buf488 = reinterpret_tensor(buf486, (1, 4, 512, 512), (1048576, 262144, 512, 1), 0); del buf486  # reuse
    cpp_fused_95(c_void_p(buf488.data_ptr()), c_void_p(getitem_149.data_ptr()), c_void_p(alias_default_23.data_ptr()), c_void_p(buf487.data_ptr()))
    del alias_default_23
    del buf487
    del getitem_149
    buf489 = reinterpret_tensor(buf482, (4, 64, 512), (32768, 512, 1), 0); del buf482  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_69, reinterpret_tensor(buf488, (4, 512, 512), (262144, 512, 1), 0), out=buf489)
    del permute_default_69
    buf490 = reinterpret_tensor(buf465, (4, 512, 64), (32768, 64, 1), 0); del buf465  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf488, (4, 512, 512), (262144, 512, 1), 0), permute_default_70, out=buf490)
    del buf488
    del permute_default_70
    buf491 = buf455; del buf455  # reuse
    cpp_fused_view_96(c_void_p(buf485.data_ptr()), c_void_p(buf491.data_ptr()))
    buf492 = reinterpret_tensor(buf485, (512, 256), (256, 1), 0); del buf485  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf491, permute_521, out=buf492)
    del permute_521
    buf493 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf491, (256, 512), (1, 256), 0), view_2, out=buf493)
    buf494 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf495 = reinterpret_tensor(buf489, (512, 256), (1, 512), 0); del buf489  # reuse
    cpp_fused_sum_view_97(c_void_p(buf495.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf494.data_ptr()))
    buf496 = buf491; del buf491  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf495, permute_526, out=buf496)
    del permute_526
    buf497 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf495, (256, 512), (512, 1), 0), view_2, out=buf497)
    buf498 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf499 = buf451; del buf451  # reuse
    cpp_fused_sum_view_98(c_void_p(buf495.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf499.data_ptr()))
    del buf490
    buf500 = reinterpret_tensor(buf495, (512, 256), (256, 1), 0); del buf495  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf499, permute_530, out=buf500)
    del permute_530
    buf501 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf499, (256, 512), (1, 256), 0), view_2, out=buf501)
    del view_2
    buf502 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf503 = buf478; del buf478  # reuse
    cpp_fused_add_sum_99(c_void_p(buf503.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf502.data_ptr()))
    del buf492
    del buf496
    del buf499
    del buf500
    buf504 = empty((512, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf503, (512, 256), (256, 1), 0), permute_534, out=buf504)
    del permute_534
    buf505 = empty((256, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf503, (256, 512), (1, 256), 0), view, out=buf505)
    del view
    buf506 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf507 = buf477; del buf477  # reuse
    buf508 = buf476; del buf476  # reuse
    buf513 = empty((1, 512, 128), device='cpu', dtype=torch.float32)
    buf517 = empty((1, 512, 128), device='cpu', dtype=torch.float32)
    buf521 = empty((1, 512, 128), device='cpu', dtype=torch.float32)
    buf510 = empty((128, ), device='cpu', dtype=torch.float32)
    buf511 = empty((128, ), device='cpu', dtype=torch.float32)
    buf512 = empty((512, 128), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_sum_100(c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(getitem_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(mul_1.data_ptr()), c_void_p(div_66.data_ptr()), c_void_p(slice_4.data_ptr()), c_void_p(expand.data_ptr()), c_void_p(primals_204.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf512.data_ptr()))
    del buf503
    del buf504
    del buf507
    del buf508
    del div_66
    del getitem_3
    del mul_1
    del primals_4
    aten.index_put_(buf512, [slice_4], buf513, True)
    del buf513
    del slice_4
    buf516 = empty((2, 128), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_101(c_void_p(buf516.data_ptr()))
    aten.index_put_(buf516, [expand], buf517, True)
    del buf517
    del expand
    buf520 = empty((30522, 128), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_102(c_void_p(buf520.data_ptr()))
    aten.index_put_(buf520, [primals_204], buf521, True)
    del buf521
    del primals_204
    return (buf520, buf516, buf512, buf510, buf511, reinterpret_tensor(buf505, (256, 128), (128, 1), 0), reinterpret_tensor(buf506, (256, ), (1, ), 0), reinterpret_tensor(buf501, (256, 256), (256, 1), 0), reinterpret_tensor(buf502, (256, ), (1, ), 0), reinterpret_tensor(buf497, (256, 256), (256, 1), 0), reinterpret_tensor(buf498, (256, ), (1, ), 0), reinterpret_tensor(buf493, (256, 256), (256, 1), 0), reinterpret_tensor(buf494, (256, ), (1, ), 0), reinterpret_tensor(buf483, (256, 256), (256, 1), 0), reinterpret_tensor(buf484, (256, ), (1, ), 0), buf479, buf480, reinterpret_tensor(buf474, (1024, 256), (256, 1), 0), reinterpret_tensor(buf475, (1024, ), (1, ), 0), reinterpret_tensor(buf470, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf471, (256, ), (1, ), 0), buf466, buf467, reinterpret_tensor(buf460, (256, 256), (256, 1), 0), reinterpret_tensor(buf461, (256, ), (1, ), 0), reinterpret_tensor(buf456, (256, 256), (256, 1), 0), reinterpret_tensor(buf457, (256, ), (1, ), 0), reinterpret_tensor(buf452, (256, 256), (256, 1), 0), reinterpret_tensor(buf453, (256, ), (1, ), 0), reinterpret_tensor(buf442, (256, 256), (256, 1), 0), reinterpret_tensor(buf443, (256, ), (1, ), 0), buf438, buf439, reinterpret_tensor(buf433, (1024, 256), (256, 1), 0), reinterpret_tensor(buf434, (1024, ), (1, ), 0), reinterpret_tensor(buf429, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf430, (256, ), (1, ), 0), buf425, buf426, reinterpret_tensor(buf419, (256, 256), (256, 1), 0), reinterpret_tensor(buf420, (256, ), (1, ), 0), reinterpret_tensor(buf415, (256, 256), (256, 1), 0), reinterpret_tensor(buf416, (256, ), (1, ), 0), reinterpret_tensor(buf411, (256, 256), (256, 1), 0), reinterpret_tensor(buf412, (256, ), (1, ), 0), reinterpret_tensor(buf401, (256, 256), (256, 1), 0), reinterpret_tensor(buf402, (256, ), (1, ), 0), buf397, buf398, reinterpret_tensor(buf392, (1024, 256), (256, 1), 0), reinterpret_tensor(buf393, (1024, ), (1, ), 0), reinterpret_tensor(buf388, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf389, (256, ), (1, ), 0), buf384, buf385, reinterpret_tensor(buf378, (256, 256), (256, 1), 0), reinterpret_tensor(buf379, (256, ), (1, ), 0), reinterpret_tensor(buf374, (256, 256), (256, 1), 0), reinterpret_tensor(buf375, (256, ), (1, ), 0), reinterpret_tensor(buf370, (256, 256), (256, 1), 0), reinterpret_tensor(buf371, (256, ), (1, ), 0), reinterpret_tensor(buf360, (256, 256), (256, 1), 0), reinterpret_tensor(buf361, (256, ), (1, ), 0), buf356, buf357, reinterpret_tensor(buf351, (1024, 256), (256, 1), 0), reinterpret_tensor(buf352, (1024, ), (1, ), 0), reinterpret_tensor(buf347, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf348, (256, ), (1, ), 0), buf343, buf344, reinterpret_tensor(buf337, (256, 256), (256, 1), 0), reinterpret_tensor(buf338, (256, ), (1, ), 0), reinterpret_tensor(buf333, (256, 256), (256, 1), 0), reinterpret_tensor(buf334, (256, ), (1, ), 0), reinterpret_tensor(buf329, (256, 256), (256, 1), 0), reinterpret_tensor(buf330, (256, ), (1, ), 0), reinterpret_tensor(buf319, (256, 256), (256, 1), 0), reinterpret_tensor(buf320, (256, ), (1, ), 0), buf315, buf316, reinterpret_tensor(buf310, (1024, 256), (256, 1), 0), reinterpret_tensor(buf311, (1024, ), (1, ), 0), reinterpret_tensor(buf306, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf307, (256, ), (1, ), 0), buf302, buf303, reinterpret_tensor(buf296, (256, 256), (256, 1), 0), reinterpret_tensor(buf297, (256, ), (1, ), 0), reinterpret_tensor(buf292, (256, 256), (256, 1), 0), reinterpret_tensor(buf293, (256, ), (1, ), 0), reinterpret_tensor(buf288, (256, 256), (256, 1), 0), reinterpret_tensor(buf289, (256, ), (1, ), 0), reinterpret_tensor(buf278, (256, 256), (256, 1), 0), reinterpret_tensor(buf279, (256, ), (1, ), 0), buf274, buf275, reinterpret_tensor(buf269, (1024, 256), (256, 1), 0), reinterpret_tensor(buf270, (1024, ), (1, ), 0), reinterpret_tensor(buf265, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf266, (256, ), (1, ), 0), buf261, buf262, reinterpret_tensor(buf255, (256, 256), (256, 1), 0), reinterpret_tensor(buf256, (256, ), (1, ), 0), reinterpret_tensor(buf251, (256, 256), (256, 1), 0), reinterpret_tensor(buf252, (256, ), (1, ), 0), reinterpret_tensor(buf247, (256, 256), (256, 1), 0), reinterpret_tensor(buf248, (256, ), (1, ), 0), reinterpret_tensor(buf237, (256, 256), (256, 1), 0), reinterpret_tensor(buf238, (256, ), (1, ), 0), buf233, buf234, reinterpret_tensor(buf228, (1024, 256), (256, 1), 0), reinterpret_tensor(buf229, (1024, ), (1, ), 0), reinterpret_tensor(buf224, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf225, (256, ), (1, ), 0), buf220, buf221, reinterpret_tensor(buf214, (256, 256), (256, 1), 0), reinterpret_tensor(buf215, (256, ), (1, ), 0), reinterpret_tensor(buf210, (256, 256), (256, 1), 0), reinterpret_tensor(buf211, (256, ), (1, ), 0), reinterpret_tensor(buf206, (256, 256), (256, 1), 0), reinterpret_tensor(buf207, (256, ), (1, ), 0), reinterpret_tensor(buf196, (256, 256), (256, 1), 0), reinterpret_tensor(buf197, (256, ), (1, ), 0), buf192, buf193, reinterpret_tensor(buf187, (1024, 256), (256, 1), 0), reinterpret_tensor(buf188, (1024, ), (1, ), 0), reinterpret_tensor(buf183, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf184, (256, ), (1, ), 0), buf179, buf180, reinterpret_tensor(buf173, (256, 256), (256, 1), 0), reinterpret_tensor(buf174, (256, ), (1, ), 0), reinterpret_tensor(buf169, (256, 256), (256, 1), 0), reinterpret_tensor(buf170, (256, ), (1, ), 0), reinterpret_tensor(buf165, (256, 256), (256, 1), 0), reinterpret_tensor(buf166, (256, ), (1, ), 0), reinterpret_tensor(buf155, (256, 256), (256, 1), 0), reinterpret_tensor(buf156, (256, ), (1, ), 0), buf151, buf152, reinterpret_tensor(buf146, (1024, 256), (256, 1), 0), reinterpret_tensor(buf147, (1024, ), (1, ), 0), reinterpret_tensor(buf142, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf143, (256, ), (1, ), 0), buf138, buf139, reinterpret_tensor(buf132, (256, 256), (256, 1), 0), reinterpret_tensor(buf133, (256, ), (1, ), 0), reinterpret_tensor(buf128, (256, 256), (256, 1), 0), reinterpret_tensor(buf129, (256, ), (1, ), 0), reinterpret_tensor(buf124, (256, 256), (256, 1), 0), reinterpret_tensor(buf125, (256, ), (1, ), 0), reinterpret_tensor(buf114, (256, 256), (256, 1), 0), reinterpret_tensor(buf115, (256, ), (1, ), 0), buf110, buf111, reinterpret_tensor(buf105, (1024, 256), (256, 1), 0), reinterpret_tensor(buf106, (1024, ), (1, ), 0), reinterpret_tensor(buf101, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf102, (256, ), (1, ), 0), buf97, buf98, reinterpret_tensor(buf91, (256, 256), (256, 1), 0), reinterpret_tensor(buf92, (256, ), (1, ), 0), reinterpret_tensor(buf87, (256, 256), (256, 1), 0), reinterpret_tensor(buf88, (256, ), (1, ), 0), reinterpret_tensor(buf83, (256, 256), (256, 1), 0), reinterpret_tensor(buf84, (256, ), (1, ), 0), reinterpret_tensor(buf73, (256, 256), (256, 1), 0), reinterpret_tensor(buf74, (256, ), (1, ), 0), buf69, buf70, reinterpret_tensor(buf64, (1024, 256), (256, 1), 0), reinterpret_tensor(buf65, (1024, ), (1, ), 0), reinterpret_tensor(buf60, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf61, (256, ), (1, ), 0), buf56, buf57, reinterpret_tensor(buf50, (256, 256), (256, 1), 0), reinterpret_tensor(buf51, (256, ), (1, ), 0), reinterpret_tensor(buf46, (256, 256), (256, 1), 0), reinterpret_tensor(buf47, (256, ), (1, ), 0), reinterpret_tensor(buf42, (256, 256), (256, 1), 0), reinterpret_tensor(buf43, (256, ), (1, ), 0), reinterpret_tensor(buf32, (256, 256), (256, 1), 0), reinterpret_tensor(buf33, (256, ), (1, ), 0), buf28, buf29, reinterpret_tensor(buf23, (1024, 256), (256, 1), 0), reinterpret_tensor(buf24, (1024, ), (1, ), 0), reinterpret_tensor(buf19, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf20, (256, ), (1, ), 0), buf15, buf16, reinterpret_tensor(buf10, (2, 256), (256, 1), 0), reinterpret_tensor(buf11, (2, ), (1, ), 0), None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    expand = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    slice_4 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    mul_1 = rand_strided((1, 512, 128), (65536, 128, 1), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((1, 512, 128), (65536, 128, 1), device='cpu', dtype=torch.bool)
    view = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    view_2 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_149 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_67 = rand_strided((4, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_68 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_23 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_69 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_70 = rand_strided((4, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_18 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_3 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_20 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_5 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_22 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_11 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_8 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_24 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_147 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_61 = rand_strided((4, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_62 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_21 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_63 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_64 = rand_strided((4, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_40 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_17 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_10 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_42 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_11 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_44 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_21 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_15 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_46 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_145 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_55 = rand_strided((4, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_56 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_19 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_57 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_58 = rand_strided((4, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_62 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_27 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_17 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_64 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_17 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_66 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_31 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_22 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_68 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_143 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_49 = rand_strided((4, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_50 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_17 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_51 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_52 = rand_strided((4, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_84 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_37 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_24 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_86 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_23 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_88 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_41 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_29 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_90 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_141 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_43 = rand_strided((4, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_44 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_15 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_45 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_46 = rand_strided((4, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_106 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_47 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_31 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_108 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_29 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_110 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_51 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_36 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_112 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_139 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_37 = rand_strided((4, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_38 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_13 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_39 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_40 = rand_strided((4, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_128 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_57 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_38 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_130 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_35 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_132 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_61 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_43 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_134 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_137 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_31 = rand_strided((4, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_32 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_11 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_33 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_34 = rand_strided((4, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_150 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_67 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_45 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_152 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_41 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_154 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_71 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_50 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_156 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_135 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_25 = rand_strided((4, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_26 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_9 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_27 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_28 = rand_strided((4, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_172 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_77 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_52 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_174 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_47 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_176 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_81 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_57 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_178 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_133 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_19 = rand_strided((4, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_20 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_7 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_21 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_22 = rand_strided((4, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_194 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_87 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_59 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_196 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_53 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_198 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_91 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_64 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_200 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_131 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_13 = rand_strided((4, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_14 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_5 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_15 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_16 = rand_strided((4, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_216 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_97 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_66 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_218 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_59 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_220 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_101 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_71 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_222 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_129 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_7 = rand_strided((4, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_8 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_3 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_9 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_10 = rand_strided((4, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_238 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_107 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_73 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_240 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_65 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_242 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_111 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_78 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_244 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_127 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_1 = rand_strided((4, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_2 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_1 = rand_strided((1, 4, 512, 512), (1048576, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_3 = rand_strided((4, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_4 = rand_strided((4, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_260 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_117 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_80 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_262 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_71 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    view_264 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    getitem_121 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.bool)
    mul_85 = rand_strided((1, 512, 256), (131072, 256, 1), device='cpu', dtype=torch.float32)
    view_266 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    sub_39 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    ne = rand_strided((1, ), (1, ), device='cpu', dtype=torch.bool)
    sub_41 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    ne_3 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.bool)
    ne_6 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.bool)
    where_4 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.int64)
    ne_8 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.bool)
    where_6 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.int64)
    permute_134 = rand_strided((2, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_30 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_138 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_142 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_31 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_146 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_158 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_163 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_167 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_33 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_171 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_175 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_34 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_179 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_191 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_196 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_200 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_36 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_204 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_208 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_37 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_212 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_224 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_229 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_233 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_39 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_237 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_241 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_40 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_245 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_257 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_262 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_266 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_42 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_270 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_274 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_43 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_278 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_290 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_295 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_299 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_45 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_303 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_307 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_46 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_311 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_323 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_328 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_332 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_48 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_336 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_340 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_49 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_344 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_356 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_361 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_365 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_51 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_369 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_373 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_52 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_377 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_389 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_394 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_398 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_54 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_402 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_406 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_55 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_410 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_422 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_427 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_431 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_57 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_435 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_439 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_58 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_443 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_455 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_460 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_464 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_60 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_468 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_472 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_61 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_476 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_488 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_493 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_497 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_63 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_501 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    permute_505 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_64 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_509 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_521 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_526 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_530 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_534 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    div_66 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    tangents_3 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_16, primals_22, primals_32, primals_38, primals_48, primals_54, primals_64, primals_70, primals_80, primals_86, primals_96, primals_102, primals_112, primals_118, primals_128, primals_134, primals_144, primals_150, primals_160, primals_166, primals_176, primals_182, primals_192, primals_198, primals_204, expand, slice_4, mul_1, getitem_3, view, view_2, getitem_149, permute_default_67, permute_default_68, alias_default_23, permute_default_69, permute_default_70, view_18, getitem_7, mul_3, view_20, addmm_5, view_22, getitem_11, mul_8, view_24, getitem_147, permute_default_61, permute_default_62, alias_default_21, permute_default_63, permute_default_64, view_40, getitem_17, mul_10, view_42, addmm_11, view_44, getitem_21, mul_15, view_46, getitem_145, permute_default_55, permute_default_56, alias_default_19, permute_default_57, permute_default_58, view_62, getitem_27, mul_17, view_64, addmm_17, view_66, getitem_31, mul_22, view_68, getitem_143, permute_default_49, permute_default_50, alias_default_17, permute_default_51, permute_default_52, view_84, getitem_37, mul_24, view_86, addmm_23, view_88, getitem_41, mul_29, view_90, getitem_141, permute_default_43, permute_default_44, alias_default_15, permute_default_45, permute_default_46, view_106, getitem_47, mul_31, view_108, addmm_29, view_110, getitem_51, mul_36, view_112, getitem_139, permute_default_37, permute_default_38, alias_default_13, permute_default_39, permute_default_40, view_128, getitem_57, mul_38, view_130, addmm_35, view_132, getitem_61, mul_43, view_134, getitem_137, permute_default_31, permute_default_32, alias_default_11, permute_default_33, permute_default_34, view_150, getitem_67, mul_45, view_152, addmm_41, view_154, getitem_71, mul_50, view_156, getitem_135, permute_default_25, permute_default_26, alias_default_9, permute_default_27, permute_default_28, view_172, getitem_77, mul_52, view_174, addmm_47, view_176, getitem_81, mul_57, view_178, getitem_133, permute_default_19, permute_default_20, alias_default_7, permute_default_21, permute_default_22, view_194, getitem_87, mul_59, view_196, addmm_53, view_198, getitem_91, mul_64, view_200, getitem_131, permute_default_13, permute_default_14, alias_default_5, permute_default_15, permute_default_16, view_216, getitem_97, mul_66, view_218, addmm_59, view_220, getitem_101, mul_71, view_222, getitem_129, permute_default_7, permute_default_8, alias_default_3, permute_default_9, permute_default_10, view_238, getitem_107, mul_73, view_240, addmm_65, view_242, getitem_111, mul_78, view_244, getitem_127, permute_default_1, permute_default_2, alias_default_1, permute_default_3, permute_default_4, view_260, getitem_117, mul_80, view_262, addmm_71, view_264, getitem_121, mul_85, view_266, sub_39, ne, sub_41, ne_3, ne_6, where_4, ne_8, where_6, permute_134, div_30, permute_138, permute_142, div_31, permute_146, permute_158, permute_163, permute_167, div_33, permute_171, permute_175, div_34, permute_179, permute_191, permute_196, permute_200, div_36, permute_204, permute_208, div_37, permute_212, permute_224, permute_229, permute_233, div_39, permute_237, permute_241, div_40, permute_245, permute_257, permute_262, permute_266, div_42, permute_270, permute_274, div_43, permute_278, permute_290, permute_295, permute_299, div_45, permute_303, permute_307, div_46, permute_311, permute_323, permute_328, permute_332, div_48, permute_336, permute_340, div_49, permute_344, permute_356, permute_361, permute_365, div_51, permute_369, permute_373, div_52, permute_377, permute_389, permute_394, permute_398, div_54, permute_402, permute_406, div_55, permute_410, permute_422, permute_427, permute_431, div_57, permute_435, permute_439, div_58, permute_443, permute_455, permute_460, permute_464, div_60, permute_468, permute_472, div_61, permute_476, permute_488, permute_493, permute_497, div_63, permute_501, permute_505, div_64, permute_509, permute_521, permute_526, permute_530, permute_534, div_66, tangents_1, tangents_2, tangents_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('ElectraForQuestionAnswering', benchmark_compiled_module)
