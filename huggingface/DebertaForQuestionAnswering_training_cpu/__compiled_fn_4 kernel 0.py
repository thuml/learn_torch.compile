
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


cpp_fused__log_softmax_backward_data_cat_div_masked_fill_nll_loss_backward_nll_loss_forward_2 = async_compile.cpp('''
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
    auto in_ptr1 = in_out_ptr0;
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
                        auto tmp2 = in_ptr3[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp3 = tmp2.neg();
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp8 = tmp7 / tmp6;
                        auto tmp9 = tmp3 * tmp8;
                        auto tmp10 = tmp2 / tmp6;
                        auto tmp11 = tmp10.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = in_ptr3[static_cast<long>(x0)];
                    auto tmp6 = out_ptr3[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp18 = out_ptr4[static_cast<long>(x0)];
                    auto tmp19 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 / tmp4;
                    auto tmp7 = static_cast<float>(2.0);
                    auto tmp8 = decltype(tmp3)(tmp3 * tmp7);
                    auto tmp9 = tmp6 / tmp8;
                    auto tmp10 = static_cast<float>(768.0);
                    auto tmp11 = tmp9 / tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp7);
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp15 = at::vec::Vectorized<float>(tmp11);
                    auto tmp16 = tmp15 * tmp14;
                    auto tmp17 = tmp5 + tmp16;
                    auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                    auto tmp21 = tmp20 / tmp10;
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp17 + tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_5 = async_compile.cpp('''
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
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
    }
}
''')


cpp_fused__softmax_backward_data_masked_fill_mul_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_10 = async_compile.cpp('''
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_11 = async_compile.cpp('''
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused__softmax_backward_data_masked_fill_mul_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_17 = async_compile.cpp('''
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused__softmax_backward_data_masked_fill_mul_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_23 = async_compile.cpp('''
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused__softmax_backward_data_masked_fill_mul_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_29 = async_compile.cpp('''
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
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
    }
}
''')


cpp_fused__softmax_backward_data_masked_fill_mul_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_35 = async_compile.cpp('''
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused__softmax_backward_data_masked_fill_mul_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_40 = async_compile.cpp('''
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_41 = async_compile.cpp('''
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused__softmax_backward_data_masked_fill_mul_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_47 = async_compile.cpp('''
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused__softmax_backward_data_masked_fill_mul_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_53 = async_compile.cpp('''
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused__softmax_backward_data_masked_fill_mul_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_58 = async_compile.cpp('''
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_59 = async_compile.cpp('''
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused__softmax_backward_data_masked_fill_mul_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_65 = async_compile.cpp('''
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_66 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused__softmax_backward_data_masked_fill_mul_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_68 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
    auto in_ptr0 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr3[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x0)];
                        auto tmp1 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = out_ptr2[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr3[static_cast<long>(x0)];
                    auto tmp21 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr5 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr5 + static_cast<long>(x0));
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


cpp_fused_add_div_masked_fill_mul_neg_pow_sum_71 = async_compile.cpp('''
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = in_ptr4[static_cast<long>(x1)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        auto tmp5 = tmp4.neg();
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 / tmp8;
                        auto tmp10 = tmp9 / tmp8;
                        auto tmp11 = tmp5 * tmp10;
                        auto tmp12 = tmp4 / tmp8;
                        auto tmp13 = tmp12.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        tmp_acc1_vec = tmp_acc1_vec + tmp13;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp5 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp20 = out_ptr4[static_cast<long>(x0)];
                    auto tmp21 = out_ptr5[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp9 = static_cast<float>(2.0);
                    auto tmp10 = decltype(tmp5)(tmp5 * tmp9);
                    auto tmp11 = tmp8 / tmp10;
                    auto tmp12 = static_cast<float>(768.0);
                    auto tmp13 = tmp11 / tmp12;
                    auto tmp15 = at::vec::Vectorized<float>(tmp9);
                    auto tmp16 = tmp14 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp7 + tmp18;
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    auto tmp23 = tmp22 / tmp12;
                    auto tmp24 = at::vec::Vectorized<float>(tmp23);
                    auto tmp25 = tmp19 + tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = flag_to_float_vec(in_ptr6 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                auto tmp5 = static_cast<float>(1.1111111111111112);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                tmp7.store(out_ptr6 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
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


cpp_fused__softmax_backward_data_masked_fill_mul_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp5 = static_cast<float>(1.1111111111111112);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp5 = static_cast<float>(1.1111111111111112);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp9 - tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_sqrt_sum_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x2) + (32768L*x0)));
                            auto tmp1 = static_cast<float>(8.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (32768L*x0))];
                            auto tmp7 = static_cast<float>(8.0);
                            auto tmp8 = tmp6 / tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<long>(128);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = tmp10 & tmp12;
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr2[static_cast<long>((-32768L) + x1 + (512L*x2) + (32768L*x0))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp0 >= tmp11;
                        auto tmp18 = static_cast<long>(192);
                        auto tmp19 = tmp0 < tmp18;
                        auto tmp20 = [&]
                        {
                            auto tmp21 = in_ptr0[static_cast<long>((-128L) + x2 + (64L*x1) + (32768L*x0))];
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                        auto tmp23 = tmp13 ? tmp16 : tmp22;
                        auto tmp24 = tmp4 ? tmp9 : tmp23;
                        out_ptr2[static_cast<long>(x2 + (192L*x0) + (2304L*x1))] = tmp24;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sum_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const long* in_ptr6,
                       const long* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8)
{
    auto out_ptr2 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp11 = in_ptr4[static_cast<long>(x1)];
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
                        auto tmp7 = static_cast<float>(1.1111111111111112);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        auto tmp14 = tmp9 * tmp13;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                        tmp_acc1_vec = tmp_acc1_vec + tmp14;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp12 = in_ptr4[static_cast<long>(x0)];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = decltype(tmp5)::blendv(tmp3, tmp5, tmp0);
                        auto tmp7 = static_cast<float>(1.1111111111111112);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp11 = tmp9 * tmp10;
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 / tmp13;
                        auto tmp15 = tmp11.neg();
                        auto tmp17 = tmp16 / tmp13;
                        auto tmp18 = tmp17 / tmp13;
                        auto tmp19 = tmp15 * tmp18;
                        tmp14.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp19;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = tmp0.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x0)];
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = static_cast<float>(2.0);
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = tmp0 / tmp3;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp2);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp10 = at::vec::Vectorized<float>(tmp6);
                        auto tmp11 = tmp10 * tmp9;
                        auto tmp12 = tmp11.neg();
                        tmp_acc0_vec = tmp_acc0_vec + tmp12;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr3[static_cast<long>(x0)];
                    auto tmp2 = in_ptr4[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp14 = out_ptr4[static_cast<long>(x0)];
                    auto tmp15 = out_ptr5[static_cast<long>(x0)];
                    auto tmp20 = in_ptr6[static_cast<long>(x0)];
                    auto tmp27 = in_ptr7[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(2.0);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = tmp1 / tmp4;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp9 = at::vec::Vectorized<float>(tmp3);
                    auto tmp10 = tmp8 * tmp9;
                    auto tmp11 = at::vec::Vectorized<float>(tmp7);
                    auto tmp12 = tmp11 * tmp10;
                    auto tmp13 = tmp0 + tmp12;
                    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                    auto tmp17 = tmp16 / tmp6;
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp13 + tmp18;
                    auto tmp21 = static_cast<int>(-1);
                    auto tmp22 = tmp20 == tmp21;
                    auto tmp23 = static_cast<float>(0.0);
                    auto tmp24 = to_float_mask(tmp22);
                    auto tmp25 = at::vec::Vectorized<float>(tmp23);
                    auto tmp26 = decltype(tmp25)::blendv(tmp19, tmp25, tmp24);
                    auto tmp28 = static_cast<int>(0);
                    auto tmp29 = tmp27 == tmp28;
                    auto tmp30 = to_float_mask(tmp29);
                    auto tmp31 = decltype(tmp25)::blendv(tmp19, tmp25, tmp30);
                    tmp26.store(out_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    tmp31.store(out_ptr7 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr8 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(38603520L); x0+=static_cast<long>(8L))
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
    primals_1, primals_5, primals_7, primals_11, primals_13, primals_17, primals_19, primals_23, primals_25, primals_29, primals_31, primals_35, primals_37, primals_41, primals_43, primals_47, primals_49, primals_53, primals_55, primals_59, primals_61, primals_65, primals_67, primals_71, primals_73, primals_164, slice_1, sub, sqrt, convert_element_type, view, convert_element_type_2, view_12, convert_element_type_3, sub_6, sqrt_2, view_14, addmm_1, view_16, convert_element_type_4, sub_9, sqrt_3, view_18, convert_element_type_6, view_30, convert_element_type_7, sub_14, sqrt_5, view_32, addmm_4, view_34, convert_element_type_8, sub_17, sqrt_6, view_36, convert_element_type_10, view_48, convert_element_type_11, sub_22, sqrt_8, view_50, addmm_7, view_52, convert_element_type_12, sub_25, sqrt_9, view_54, convert_element_type_14, view_66, convert_element_type_15, sub_30, sqrt_11, view_68, addmm_10, view_70, convert_element_type_16, sub_33, sqrt_12, view_72, convert_element_type_18, view_84, convert_element_type_19, sub_38, sqrt_14, view_86, addmm_13, view_88, convert_element_type_20, sub_41, sqrt_15, view_90, convert_element_type_22, view_102, convert_element_type_23, sub_46, sqrt_17, view_104, addmm_16, view_106, convert_element_type_24, sub_49, sqrt_18, view_108, convert_element_type_26, view_120, convert_element_type_27, sub_54, sqrt_20, view_122, addmm_19, view_124, convert_element_type_28, sub_57, sqrt_21, view_126, convert_element_type_30, view_138, convert_element_type_31, sub_62, sqrt_23, view_140, addmm_22, view_142, convert_element_type_32, sub_65, sqrt_24, view_144, convert_element_type_34, view_156, convert_element_type_35, sub_70, sqrt_26, view_158, addmm_25, view_160, convert_element_type_36, sub_73, sqrt_27, view_162, convert_element_type_38, view_174, convert_element_type_39, sub_78, sqrt_29, view_176, addmm_28, view_178, convert_element_type_40, sub_81, sqrt_30, view_180, convert_element_type_42, view_192, convert_element_type_43, sub_86, sqrt_32, view_194, addmm_31, view_196, convert_element_type_44, sub_89, sqrt_33, view_198, convert_element_type_46, view_210, convert_element_type_47, sub_94, sqrt_35, view_212, addmm_34, view_214, convert_element_type_48, sub_97, sqrt_36, view_216, sub_100, ne, sub_102, ne_3, ne_6, where_65, ne_8, where_67, permute_146, permute_150, permute_154, permute_158, permute_163, permute_164, alias_45, permute_165, permute_166, permute_173, permute_175, permute_179, permute_183, permute_188, permute_189, alias_50, permute_190, permute_191, permute_198, permute_200, permute_204, permute_208, permute_213, permute_214, alias_55, permute_215, permute_216, permute_223, permute_225, permute_229, permute_233, permute_238, permute_239, alias_60, permute_240, permute_241, permute_248, permute_250, permute_254, permute_258, permute_263, permute_264, alias_65, permute_265, permute_266, permute_273, permute_275, permute_279, permute_283, permute_288, permute_289, alias_70, permute_290, permute_291, permute_298, permute_300, permute_304, permute_308, permute_313, permute_314, alias_75, permute_315, permute_316, permute_323, permute_325, permute_329, permute_333, permute_338, permute_339, alias_80, permute_340, permute_341, permute_348, permute_350, permute_354, permute_358, permute_363, permute_364, alias_85, permute_365, permute_366, permute_373, permute_375, permute_379, permute_383, permute_388, permute_389, alias_90, permute_390, permute_391, permute_398, permute_400, permute_404, permute_408, permute_413, permute_414, alias_95, permute_415, permute_416, permute_423, permute_425, permute_429, permute_433, permute_438, permute_439, alias_100, permute_440, permute_441, permute_448, tangents_1, tangents_2, tangents_3 = args
    args.clear()
    assert_size_stride(primals_1, (768, ), (1, ))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_164, (1, 512), (512, 1))
    assert_size_stride(slice_1, (1, 512), (512, 1))
    assert_size_stride(sub, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt, (1, 512, 1), (512, 1, 1))
    assert_size_stride(convert_element_type, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_2, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_12, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_3, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_6, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_2, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_14, (512, 768), (768, 1))
    assert_size_stride(addmm_1, (512, 3072), (3072, 1))
    assert_size_stride(view_16, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_4, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_9, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_3, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_18, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_6, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_30, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_7, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_14, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_5, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_32, (512, 768), (768, 1))
    assert_size_stride(addmm_4, (512, 3072), (3072, 1))
    assert_size_stride(view_34, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_8, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_17, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_6, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_36, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_10, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_48, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_11, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_22, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_8, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_50, (512, 768), (768, 1))
    assert_size_stride(addmm_7, (512, 3072), (3072, 1))
    assert_size_stride(view_52, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_12, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_25, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_9, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_54, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_14, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_66, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_15, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_30, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_11, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_68, (512, 768), (768, 1))
    assert_size_stride(addmm_10, (512, 3072), (3072, 1))
    assert_size_stride(view_70, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_16, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_33, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_12, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_72, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_18, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_84, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_19, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_38, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_14, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_86, (512, 768), (768, 1))
    assert_size_stride(addmm_13, (512, 3072), (3072, 1))
    assert_size_stride(view_88, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_20, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_41, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_15, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_90, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_22, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_102, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_23, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_46, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_17, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_104, (512, 768), (768, 1))
    assert_size_stride(addmm_16, (512, 3072), (3072, 1))
    assert_size_stride(view_106, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_24, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_49, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_18, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_108, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_26, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_120, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_27, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_54, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_20, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_122, (512, 768), (768, 1))
    assert_size_stride(addmm_19, (512, 3072), (3072, 1))
    assert_size_stride(view_124, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_28, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_57, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_21, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_126, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_30, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_138, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_31, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_62, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_23, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_140, (512, 768), (768, 1))
    assert_size_stride(addmm_22, (512, 3072), (3072, 1))
    assert_size_stride(view_142, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_32, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_65, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_24, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_144, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_34, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_156, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_35, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_70, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_26, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_158, (512, 768), (768, 1))
    assert_size_stride(addmm_25, (512, 3072), (3072, 1))
    assert_size_stride(view_160, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_36, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_73, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_27, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_162, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_38, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_174, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_39, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_78, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_29, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_176, (512, 768), (768, 1))
    assert_size_stride(addmm_28, (512, 3072), (3072, 1))
    assert_size_stride(view_178, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_40, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_81, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_30, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_180, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_42, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_192, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_43, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_86, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_32, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_194, (512, 768), (768, 1))
    assert_size_stride(addmm_31, (512, 3072), (3072, 1))
    assert_size_stride(view_196, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_44, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_89, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_33, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_198, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_46, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(view_210, (512, 768), (768, 1))
    assert_size_stride(convert_element_type_47, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_94, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_35, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_212, (512, 768), (768, 1))
    assert_size_stride(addmm_34, (512, 3072), (3072, 1))
    assert_size_stride(view_214, (512, 3072), (3072, 1))
    assert_size_stride(convert_element_type_48, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sub_97, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(sqrt_36, (1, 512, 1), (512, 1, 1))
    assert_size_stride(view_216, (512, 768), (768, 1))
    assert_size_stride(sub_100, (1, 512), (512, 1))
    assert_size_stride(ne, (1, ), (1, ))
    assert_size_stride(sub_102, (1, 512), (512, 1))
    assert_size_stride(ne_3, (1, ), (1, ))
    assert_size_stride(ne_6, (1, 1), (1, 1))
    assert_size_stride(where_65, (1, 1), (1, 1))
    assert_size_stride(ne_8, (1, 1), (1, 1))
    assert_size_stride(where_67, (1, 1), (1, 1))
    assert_size_stride(permute_146, (2, 768), (768, 1))
    assert_size_stride(permute_150, (768, 3072), (3072, 1))
    assert_size_stride(permute_154, (3072, 768), (768, 1))
    assert_size_stride(permute_158, (768, 768), (768, 1))
    assert_size_stride(permute_163, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_164, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_45, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_165, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_166, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_173, (2304, 768), (768, 1))
    assert_size_stride(permute_175, (768, 3072), (3072, 1))
    assert_size_stride(permute_179, (3072, 768), (768, 1))
    assert_size_stride(permute_183, (768, 768), (768, 1))
    assert_size_stride(permute_188, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_189, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_50, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_190, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_191, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_198, (2304, 768), (768, 1))
    assert_size_stride(permute_200, (768, 3072), (3072, 1))
    assert_size_stride(permute_204, (3072, 768), (768, 1))
    assert_size_stride(permute_208, (768, 768), (768, 1))
    assert_size_stride(permute_213, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_214, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_55, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_215, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_216, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_223, (2304, 768), (768, 1))
    assert_size_stride(permute_225, (768, 3072), (3072, 1))
    assert_size_stride(permute_229, (3072, 768), (768, 1))
    assert_size_stride(permute_233, (768, 768), (768, 1))
    assert_size_stride(permute_238, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_239, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_60, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_240, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_241, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_248, (2304, 768), (768, 1))
    assert_size_stride(permute_250, (768, 3072), (3072, 1))
    assert_size_stride(permute_254, (3072, 768), (768, 1))
    assert_size_stride(permute_258, (768, 768), (768, 1))
    assert_size_stride(permute_263, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_264, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_65, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_265, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_266, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_273, (2304, 768), (768, 1))
    assert_size_stride(permute_275, (768, 3072), (3072, 1))
    assert_size_stride(permute_279, (3072, 768), (768, 1))
    assert_size_stride(permute_283, (768, 768), (768, 1))
    assert_size_stride(permute_288, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_289, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_70, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_290, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_291, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_298, (2304, 768), (768, 1))
    assert_size_stride(permute_300, (768, 3072), (3072, 1))
    assert_size_stride(permute_304, (3072, 768), (768, 1))
    assert_size_stride(permute_308, (768, 768), (768, 1))
    assert_size_stride(permute_313, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_314, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_75, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_315, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_316, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_323, (2304, 768), (768, 1))
    assert_size_stride(permute_325, (768, 3072), (3072, 1))
    assert_size_stride(permute_329, (3072, 768), (768, 1))
    assert_size_stride(permute_333, (768, 768), (768, 1))
    assert_size_stride(permute_338, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_339, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_80, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_340, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_341, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_348, (2304, 768), (768, 1))
    assert_size_stride(permute_350, (768, 3072), (3072, 1))
    assert_size_stride(permute_354, (3072, 768), (768, 1))
    assert_size_stride(permute_358, (768, 768), (768, 1))
    assert_size_stride(permute_363, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_364, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_85, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_365, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_366, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_373, (2304, 768), (768, 1))
    assert_size_stride(permute_375, (768, 3072), (3072, 1))
    assert_size_stride(permute_379, (3072, 768), (768, 1))
    assert_size_stride(permute_383, (768, 768), (768, 1))
    assert_size_stride(permute_388, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_389, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_90, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_390, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_391, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_398, (2304, 768), (768, 1))
    assert_size_stride(permute_400, (768, 3072), (3072, 1))
    assert_size_stride(permute_404, (3072, 768), (768, 1))
    assert_size_stride(permute_408, (768, 768), (768, 1))
    assert_size_stride(permute_413, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_414, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_95, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_415, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_416, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_423, (2304, 768), (768, 1))
    assert_size_stride(permute_425, (768, 3072), (3072, 1))
    assert_size_stride(permute_429, (3072, 768), (768, 1))
    assert_size_stride(permute_433, (768, 768), (768, 1))
    assert_size_stride(permute_438, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_439, (12, 64, 512), (64, 1, 768))
    assert_size_stride(alias_100, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_440, (12, 64, 512), (64, 1, 768))
    assert_size_stride(permute_441, (12, 512, 64), (192, 2304, 1))
    assert_size_stride(permute_448, (2304, 768), (768, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 512), (512, 1))
    assert_size_stride(tangents_3, (1, 512), (512, 1))
    buf0 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_nll_loss_backward_0(c_void_p(buf0.data_ptr()))
    aten.scatter_(buf0,1,where_65,-1.0)
    del where_65
    buf4 = empty((1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_nll_loss_backward_1(c_void_p(buf4.data_ptr()))
    aten.scatter_(buf4,1,where_67,-1.0)
    del where_67
    buf3 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf7 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf8 = empty((1, 512, 2), device='cpu', dtype=torch.float32)
    cpp_fused__log_softmax_backward_data_cat_div_masked_fill_nll_loss_backward_nll_loss_forward_2(c_void_p(buf0.data_ptr()), c_void_p(ne_6.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(ne_3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(ne_8.data_ptr()), c_void_p(ne.data_ptr()), c_void_p(tangents_2.data_ptr()), c_void_p(sub_100.data_ptr()), c_void_p(tangents_3.data_ptr()), c_void_p(sub_102.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()))
    del buf3
    del buf7
    del ne
    del ne_3
    del ne_6
    del ne_8
    del sub_100
    del sub_102
    del tangents_1
    del tangents_2
    del tangents_3
    buf9 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf8, (512, 2), (2, 1), 0), permute_146, out=buf9)
    del permute_146
    buf10 = empty((2, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf8, (2, 512), (1, 2), 0), view_216, out=buf10)
    del view_216
    buf11 = empty((1, 2), device='cpu', dtype=torch.float32)
    buf12 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf13 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf14 = reinterpret_tensor(buf4, (1, 512, 1), (512, 1, 512), 0); del buf4  # reuse
    buf15 = reinterpret_tensor(buf0, (1, 512, 1), (512, 1, 512), 0); del buf0  # reuse
    buf16 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf17 = reinterpret_tensor(buf9, (1, 512, 768), (393216, 768, 1), 0); del buf9  # reuse
    buf18 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_3(c_void_p(buf17.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(sub_97.data_ptr()), c_void_p(sqrt_36.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(convert_element_type_48.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf18.data_ptr()))
    del buf8
    del convert_element_type_48
    del primals_73
    del sqrt_36
    del sub_97
    buf19 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf18, (512, 768), (768, 1), 0), permute_150, out=buf19)
    del permute_150
    buf20 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf18, (768, 512), (1, 768), 0), view_214, out=buf20)
    del view_214
    buf21 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf22 = reinterpret_tensor(buf19, (1, 512, 3072), (1572864, 3072, 1), 0); del buf19  # reuse
    cpp_fused_gelu_gelu_backward_sum_4(c_void_p(buf22.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(buf21.data_ptr()))
    del addmm_34
    buf23 = reinterpret_tensor(buf18, (512, 768), (768, 1), 0); del buf18  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf22, (512, 3072), (3072, 1), 0), permute_154, out=buf23)
    del permute_154
    buf24 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf22, (3072, 512), (1, 3072), 0), view_212, out=buf24)
    del view_212
    buf25 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf26 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf27 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf28 = buf16; del buf16  # reuse
    buf29 = buf15; del buf15  # reuse
    buf30 = buf14; del buf14  # reuse
    buf31 = buf17; del buf17  # reuse
    buf32 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_5(c_void_p(buf31.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(sub_94.data_ptr()), c_void_p(sqrt_35.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(convert_element_type_47.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf32.data_ptr()))
    del convert_element_type_47
    del primals_71
    del sqrt_35
    del sub_94
    buf33 = buf23; del buf23  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf32, (512, 768), (768, 1), 0), permute_158, out=buf33)
    del permute_158
    buf34 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf32, (768, 512), (1, 768), 0), view_210, out=buf34)
    del view_210
    buf35 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_6(c_void_p(buf32.data_ptr()), c_void_p(buf35.data_ptr()))
    buf36 = reinterpret_tensor(buf32, (12, 512, 64), (32768, 64, 1), 0); del buf32  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_163, reinterpret_tensor(buf33, (12, 512, 64), (64, 768, 1), 0), out=buf36)
    del permute_163
    buf37 = empty((12, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf33, (12, 512, 64), (64, 768, 1), 0), permute_164, out=buf37)
    del permute_164
    buf38 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    buf39 = reinterpret_tensor(buf37, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf37  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_7(c_void_p(buf39.data_ptr()), c_void_p(convert_element_type_46.data_ptr()), c_void_p(alias_45.data_ptr()), c_void_p(buf38.data_ptr()))
    del alias_45
    del convert_element_type_46
    buf40 = reinterpret_tensor(buf33, (12, 64, 512), (32768, 512, 1), 0); del buf33  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_165, reinterpret_tensor(buf39, (12, 512, 512), (262144, 512, 1), 0), out=buf40)
    del permute_165
    buf41 = empty((12, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf39, (12, 512, 512), (262144, 512, 1), 0), permute_166, out=buf41)
    del permute_166
    buf42 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf43 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf44 = empty((1, 512, 12, 192), device='cpu', dtype=torch.float32)
    cpp_fused_clone_div_sqrt_sum_8(c_void_p(buf36.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()))
    buf45 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf44, (2304, 512), (1, 2304), 0), view_198, out=buf45)
    del view_198
    buf46 = reinterpret_tensor(buf41, (512, 768), (768, 1), 0); del buf41  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf44, (512, 2304), (2304, 1), 0), permute_173, out=buf46)
    del permute_173
    buf47 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf48 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf49 = buf30; del buf30  # reuse
    buf50 = buf29; del buf29  # reuse
    buf51 = buf28; del buf28  # reuse
    buf52 = buf31; del buf31  # reuse
    buf53 = reinterpret_tensor(buf40, (1, 512, 768), (393216, 768, 1), 0); del buf40  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_9(c_void_p(buf52.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(sub_89.data_ptr()), c_void_p(sqrt_33.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(convert_element_type_44.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()))
    del convert_element_type_44
    del primals_67
    del sqrt_33
    del sub_89
    buf54 = reinterpret_tensor(buf22, (512, 3072), (3072, 1), 0); del buf22  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf53, (512, 768), (768, 1), 0), permute_175, out=buf54)
    del permute_175
    buf55 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf53, (768, 512), (1, 768), 0), view_196, out=buf55)
    del view_196
    buf56 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf57 = reinterpret_tensor(buf54, (1, 512, 3072), (1572864, 3072, 1), 0); del buf54  # reuse
    cpp_fused_gelu_gelu_backward_sum_10(c_void_p(buf57.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(addmm_31.data_ptr()), c_void_p(buf56.data_ptr()))
    del addmm_31
    buf58 = reinterpret_tensor(buf53, (512, 768), (768, 1), 0); del buf53  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf57, (512, 3072), (3072, 1), 0), permute_179, out=buf58)
    del permute_179
    buf59 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf57, (3072, 512), (1, 3072), 0), view_194, out=buf59)
    del view_194
    buf60 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf61 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf62 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf63 = buf51; del buf51  # reuse
    buf64 = buf50; del buf50  # reuse
    buf65 = buf49; del buf49  # reuse
    buf66 = buf52; del buf52  # reuse
    buf67 = reinterpret_tensor(buf46, (1, 512, 768), (393216, 768, 1), 0); del buf46  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_11(c_void_p(buf66.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(sub_86.data_ptr()), c_void_p(sqrt_32.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(convert_element_type_43.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf67.data_ptr()))
    del convert_element_type_43
    del primals_65
    del sqrt_32
    del sub_86
    buf68 = buf58; del buf58  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf67, (512, 768), (768, 1), 0), permute_183, out=buf68)
    del permute_183
    buf69 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf67, (768, 512), (1, 768), 0), view_192, out=buf69)
    del view_192
    buf70 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_12(c_void_p(buf67.data_ptr()), c_void_p(buf70.data_ptr()))
    buf71 = reinterpret_tensor(buf67, (12, 512, 64), (32768, 64, 1), 0); del buf67  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_188, reinterpret_tensor(buf68, (12, 512, 64), (64, 768, 1), 0), out=buf71)
    del permute_188
    buf72 = reinterpret_tensor(buf39, (12, 512, 512), (262144, 512, 1), 0); del buf39  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf68, (12, 512, 64), (64, 768, 1), 0), permute_189, out=buf72)
    del permute_189
    buf73 = buf38; del buf38  # reuse
    buf74 = reinterpret_tensor(buf72, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf72  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_13(c_void_p(buf74.data_ptr()), c_void_p(convert_element_type_42.data_ptr()), c_void_p(alias_50.data_ptr()), c_void_p(buf73.data_ptr()))
    del alias_50
    del convert_element_type_42
    buf75 = reinterpret_tensor(buf68, (12, 64, 512), (32768, 512, 1), 0); del buf68  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_190, reinterpret_tensor(buf74, (12, 512, 512), (262144, 512, 1), 0), out=buf75)
    del permute_190
    buf76 = buf36; del buf36  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf74, (12, 512, 512), (262144, 512, 1), 0), permute_191, out=buf76)
    del permute_191
    buf77 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf78 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf79 = buf44; del buf44  # reuse
    cpp_fused_clone_div_sqrt_sum_14(c_void_p(buf71.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    buf80 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf79, (2304, 512), (1, 2304), 0), view_180, out=buf80)
    del view_180
    buf81 = reinterpret_tensor(buf76, (512, 768), (768, 1), 0); del buf76  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf79, (512, 2304), (2304, 1), 0), permute_198, out=buf81)
    del permute_198
    buf82 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf83 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf84 = buf65; del buf65  # reuse
    buf85 = buf64; del buf64  # reuse
    buf86 = buf63; del buf63  # reuse
    buf87 = buf66; del buf66  # reuse
    buf88 = reinterpret_tensor(buf75, (1, 512, 768), (393216, 768, 1), 0); del buf75  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_15(c_void_p(buf87.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(sub_81.data_ptr()), c_void_p(sqrt_30.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(convert_element_type_40.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf88.data_ptr()))
    del convert_element_type_40
    del primals_61
    del sqrt_30
    del sub_81
    buf89 = reinterpret_tensor(buf57, (512, 3072), (3072, 1), 0); del buf57  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf88, (512, 768), (768, 1), 0), permute_200, out=buf89)
    del permute_200
    buf90 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf88, (768, 512), (1, 768), 0), view_178, out=buf90)
    del view_178
    buf91 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf92 = reinterpret_tensor(buf89, (1, 512, 3072), (1572864, 3072, 1), 0); del buf89  # reuse
    cpp_fused_gelu_gelu_backward_sum_16(c_void_p(buf92.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(addmm_28.data_ptr()), c_void_p(buf91.data_ptr()))
    del addmm_28
    buf93 = reinterpret_tensor(buf88, (512, 768), (768, 1), 0); del buf88  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf92, (512, 3072), (3072, 1), 0), permute_204, out=buf93)
    del permute_204
    buf94 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf92, (3072, 512), (1, 3072), 0), view_176, out=buf94)
    del view_176
    buf95 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf96 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf97 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf98 = buf86; del buf86  # reuse
    buf99 = buf85; del buf85  # reuse
    buf100 = buf84; del buf84  # reuse
    buf101 = buf87; del buf87  # reuse
    buf102 = reinterpret_tensor(buf81, (1, 512, 768), (393216, 768, 1), 0); del buf81  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_17(c_void_p(buf101.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(sub_78.data_ptr()), c_void_p(sqrt_29.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(convert_element_type_39.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf102.data_ptr()))
    del convert_element_type_39
    del primals_59
    del sqrt_29
    del sub_78
    buf103 = buf93; del buf93  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf102, (512, 768), (768, 1), 0), permute_208, out=buf103)
    del permute_208
    buf104 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf102, (768, 512), (1, 768), 0), view_174, out=buf104)
    del view_174
    buf105 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_18(c_void_p(buf102.data_ptr()), c_void_p(buf105.data_ptr()))
    buf106 = reinterpret_tensor(buf102, (12, 512, 64), (32768, 64, 1), 0); del buf102  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_213, reinterpret_tensor(buf103, (12, 512, 64), (64, 768, 1), 0), out=buf106)
    del permute_213
    buf107 = reinterpret_tensor(buf74, (12, 512, 512), (262144, 512, 1), 0); del buf74  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf103, (12, 512, 64), (64, 768, 1), 0), permute_214, out=buf107)
    del permute_214
    buf108 = buf73; del buf73  # reuse
    buf109 = reinterpret_tensor(buf107, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf107  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_19(c_void_p(buf109.data_ptr()), c_void_p(convert_element_type_38.data_ptr()), c_void_p(alias_55.data_ptr()), c_void_p(buf108.data_ptr()))
    del alias_55
    del convert_element_type_38
    buf110 = reinterpret_tensor(buf103, (12, 64, 512), (32768, 512, 1), 0); del buf103  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_215, reinterpret_tensor(buf109, (12, 512, 512), (262144, 512, 1), 0), out=buf110)
    del permute_215
    buf111 = buf71; del buf71  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf109, (12, 512, 512), (262144, 512, 1), 0), permute_216, out=buf111)
    del permute_216
    buf112 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf113 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf114 = buf79; del buf79  # reuse
    cpp_fused_clone_div_sqrt_sum_20(c_void_p(buf106.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()))
    buf115 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf114, (2304, 512), (1, 2304), 0), view_162, out=buf115)
    del view_162
    buf116 = reinterpret_tensor(buf111, (512, 768), (768, 1), 0); del buf111  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf114, (512, 2304), (2304, 1), 0), permute_223, out=buf116)
    del permute_223
    buf117 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf118 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf119 = buf99; del buf99  # reuse
    buf120 = buf98; del buf98  # reuse
    buf121 = buf100; del buf100  # reuse
    buf122 = buf101; del buf101  # reuse
    buf123 = reinterpret_tensor(buf110, (1, 512, 768), (393216, 768, 1), 0); del buf110  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_21(c_void_p(buf122.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(sub_73.data_ptr()), c_void_p(sqrt_27.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(convert_element_type_36.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf123.data_ptr()))
    del convert_element_type_36
    del primals_55
    del sqrt_27
    del sub_73
    buf124 = reinterpret_tensor(buf92, (512, 3072), (3072, 1), 0); del buf92  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf123, (512, 768), (768, 1), 0), permute_225, out=buf124)
    del permute_225
    buf125 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf123, (768, 512), (1, 768), 0), view_160, out=buf125)
    del view_160
    buf126 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf127 = reinterpret_tensor(buf124, (1, 512, 3072), (1572864, 3072, 1), 0); del buf124  # reuse
    cpp_fused_gelu_gelu_backward_sum_22(c_void_p(buf127.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(addmm_25.data_ptr()), c_void_p(buf126.data_ptr()))
    del addmm_25
    buf128 = reinterpret_tensor(buf123, (512, 768), (768, 1), 0); del buf123  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf127, (512, 3072), (3072, 1), 0), permute_229, out=buf128)
    del permute_229
    buf129 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf127, (3072, 512), (1, 3072), 0), view_158, out=buf129)
    del view_158
    buf130 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf131 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf132 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf133 = buf121; del buf121  # reuse
    buf134 = buf120; del buf120  # reuse
    buf135 = buf119; del buf119  # reuse
    buf136 = buf122; del buf122  # reuse
    buf137 = reinterpret_tensor(buf116, (1, 512, 768), (393216, 768, 1), 0); del buf116  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_23(c_void_p(buf136.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(sub_70.data_ptr()), c_void_p(sqrt_26.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(convert_element_type_35.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf137.data_ptr()))
    del convert_element_type_35
    del primals_53
    del sqrt_26
    del sub_70
    buf138 = buf128; del buf128  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf137, (512, 768), (768, 1), 0), permute_233, out=buf138)
    del permute_233
    buf139 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf137, (768, 512), (1, 768), 0), view_156, out=buf139)
    del view_156
    buf140 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_24(c_void_p(buf137.data_ptr()), c_void_p(buf140.data_ptr()))
    buf141 = reinterpret_tensor(buf137, (12, 512, 64), (32768, 64, 1), 0); del buf137  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_238, reinterpret_tensor(buf138, (12, 512, 64), (64, 768, 1), 0), out=buf141)
    del permute_238
    buf142 = reinterpret_tensor(buf109, (12, 512, 512), (262144, 512, 1), 0); del buf109  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf138, (12, 512, 64), (64, 768, 1), 0), permute_239, out=buf142)
    del permute_239
    buf143 = buf108; del buf108  # reuse
    buf144 = reinterpret_tensor(buf142, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf142  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_25(c_void_p(buf144.data_ptr()), c_void_p(convert_element_type_34.data_ptr()), c_void_p(alias_60.data_ptr()), c_void_p(buf143.data_ptr()))
    del alias_60
    del convert_element_type_34
    buf145 = reinterpret_tensor(buf138, (12, 64, 512), (32768, 512, 1), 0); del buf138  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_240, reinterpret_tensor(buf144, (12, 512, 512), (262144, 512, 1), 0), out=buf145)
    del permute_240
    buf146 = buf106; del buf106  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf144, (12, 512, 512), (262144, 512, 1), 0), permute_241, out=buf146)
    del permute_241
    buf147 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf148 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf149 = buf114; del buf114  # reuse
    cpp_fused_clone_div_sqrt_sum_26(c_void_p(buf141.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()))
    buf150 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf149, (2304, 512), (1, 2304), 0), view_144, out=buf150)
    del view_144
    buf151 = reinterpret_tensor(buf146, (512, 768), (768, 1), 0); del buf146  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf149, (512, 2304), (2304, 1), 0), permute_248, out=buf151)
    del permute_248
    buf152 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf153 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf154 = buf135; del buf135  # reuse
    buf155 = buf134; del buf134  # reuse
    buf156 = buf133; del buf133  # reuse
    buf157 = buf136; del buf136  # reuse
    buf158 = reinterpret_tensor(buf145, (1, 512, 768), (393216, 768, 1), 0); del buf145  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_27(c_void_p(buf157.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(sub_65.data_ptr()), c_void_p(sqrt_24.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(convert_element_type_32.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf158.data_ptr()))
    del convert_element_type_32
    del primals_49
    del sqrt_24
    del sub_65
    buf159 = reinterpret_tensor(buf127, (512, 3072), (3072, 1), 0); del buf127  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf158, (512, 768), (768, 1), 0), permute_250, out=buf159)
    del permute_250
    buf160 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf158, (768, 512), (1, 768), 0), view_142, out=buf160)
    del view_142
    buf161 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf162 = reinterpret_tensor(buf159, (1, 512, 3072), (1572864, 3072, 1), 0); del buf159  # reuse
    cpp_fused_gelu_gelu_backward_sum_28(c_void_p(buf162.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(buf161.data_ptr()))
    del addmm_22
    buf163 = reinterpret_tensor(buf158, (512, 768), (768, 1), 0); del buf158  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf162, (512, 3072), (3072, 1), 0), permute_254, out=buf163)
    del permute_254
    buf164 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf162, (3072, 512), (1, 3072), 0), view_140, out=buf164)
    del view_140
    buf165 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf166 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf167 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf168 = buf156; del buf156  # reuse
    buf169 = buf155; del buf155  # reuse
    buf170 = buf154; del buf154  # reuse
    buf171 = buf157; del buf157  # reuse
    buf172 = reinterpret_tensor(buf151, (1, 512, 768), (393216, 768, 1), 0); del buf151  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_29(c_void_p(buf171.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(sub_62.data_ptr()), c_void_p(sqrt_23.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(convert_element_type_31.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf172.data_ptr()))
    del convert_element_type_31
    del primals_47
    del sqrt_23
    del sub_62
    buf173 = buf163; del buf163  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf172, (512, 768), (768, 1), 0), permute_258, out=buf173)
    del permute_258
    buf174 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf172, (768, 512), (1, 768), 0), view_138, out=buf174)
    del view_138
    buf175 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_30(c_void_p(buf172.data_ptr()), c_void_p(buf175.data_ptr()))
    buf176 = reinterpret_tensor(buf172, (12, 512, 64), (32768, 64, 1), 0); del buf172  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_263, reinterpret_tensor(buf173, (12, 512, 64), (64, 768, 1), 0), out=buf176)
    del permute_263
    buf177 = reinterpret_tensor(buf144, (12, 512, 512), (262144, 512, 1), 0); del buf144  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf173, (12, 512, 64), (64, 768, 1), 0), permute_264, out=buf177)
    del permute_264
    buf178 = buf143; del buf143  # reuse
    buf179 = reinterpret_tensor(buf177, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf177  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_31(c_void_p(buf179.data_ptr()), c_void_p(convert_element_type_30.data_ptr()), c_void_p(alias_65.data_ptr()), c_void_p(buf178.data_ptr()))
    del alias_65
    del convert_element_type_30
    buf180 = reinterpret_tensor(buf173, (12, 64, 512), (32768, 512, 1), 0); del buf173  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_265, reinterpret_tensor(buf179, (12, 512, 512), (262144, 512, 1), 0), out=buf180)
    del permute_265
    buf181 = buf141; del buf141  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf179, (12, 512, 512), (262144, 512, 1), 0), permute_266, out=buf181)
    del permute_266
    buf182 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf183 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf184 = buf149; del buf149  # reuse
    cpp_fused_clone_div_sqrt_sum_32(c_void_p(buf176.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()))
    buf185 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf184, (2304, 512), (1, 2304), 0), view_126, out=buf185)
    del view_126
    buf186 = reinterpret_tensor(buf181, (512, 768), (768, 1), 0); del buf181  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf184, (512, 2304), (2304, 1), 0), permute_273, out=buf186)
    del permute_273
    buf187 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf188 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf189 = buf170; del buf170  # reuse
    buf190 = buf169; del buf169  # reuse
    buf191 = buf168; del buf168  # reuse
    buf192 = buf171; del buf171  # reuse
    buf193 = reinterpret_tensor(buf180, (1, 512, 768), (393216, 768, 1), 0); del buf180  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_33(c_void_p(buf192.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(sub_57.data_ptr()), c_void_p(sqrt_21.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(convert_element_type_28.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf193.data_ptr()))
    del convert_element_type_28
    del primals_43
    del sqrt_21
    del sub_57
    buf194 = reinterpret_tensor(buf162, (512, 3072), (3072, 1), 0); del buf162  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf193, (512, 768), (768, 1), 0), permute_275, out=buf194)
    del permute_275
    buf195 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf193, (768, 512), (1, 768), 0), view_124, out=buf195)
    del view_124
    buf196 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf197 = reinterpret_tensor(buf194, (1, 512, 3072), (1572864, 3072, 1), 0); del buf194  # reuse
    cpp_fused_gelu_gelu_backward_sum_34(c_void_p(buf197.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(addmm_19.data_ptr()), c_void_p(buf196.data_ptr()))
    del addmm_19
    buf198 = reinterpret_tensor(buf193, (512, 768), (768, 1), 0); del buf193  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf197, (512, 3072), (3072, 1), 0), permute_279, out=buf198)
    del permute_279
    buf199 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf197, (3072, 512), (1, 3072), 0), view_122, out=buf199)
    del view_122
    buf200 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf201 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf202 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf203 = buf191; del buf191  # reuse
    buf204 = buf190; del buf190  # reuse
    buf205 = buf189; del buf189  # reuse
    buf206 = buf192; del buf192  # reuse
    buf207 = reinterpret_tensor(buf186, (1, 512, 768), (393216, 768, 1), 0); del buf186  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_35(c_void_p(buf206.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(sub_54.data_ptr()), c_void_p(sqrt_20.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(convert_element_type_27.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf207.data_ptr()))
    del convert_element_type_27
    del primals_41
    del sqrt_20
    del sub_54
    buf208 = buf198; del buf198  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf207, (512, 768), (768, 1), 0), permute_283, out=buf208)
    del permute_283
    buf209 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf207, (768, 512), (1, 768), 0), view_120, out=buf209)
    del view_120
    buf210 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_36(c_void_p(buf207.data_ptr()), c_void_p(buf210.data_ptr()))
    buf211 = reinterpret_tensor(buf207, (12, 512, 64), (32768, 64, 1), 0); del buf207  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_288, reinterpret_tensor(buf208, (12, 512, 64), (64, 768, 1), 0), out=buf211)
    del permute_288
    buf212 = reinterpret_tensor(buf179, (12, 512, 512), (262144, 512, 1), 0); del buf179  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf208, (12, 512, 64), (64, 768, 1), 0), permute_289, out=buf212)
    del permute_289
    buf213 = buf178; del buf178  # reuse
    buf214 = reinterpret_tensor(buf212, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf212  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_37(c_void_p(buf214.data_ptr()), c_void_p(convert_element_type_26.data_ptr()), c_void_p(alias_70.data_ptr()), c_void_p(buf213.data_ptr()))
    del alias_70
    del convert_element_type_26
    buf215 = reinterpret_tensor(buf208, (12, 64, 512), (32768, 512, 1), 0); del buf208  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_290, reinterpret_tensor(buf214, (12, 512, 512), (262144, 512, 1), 0), out=buf215)
    del permute_290
    buf216 = buf176; del buf176  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf214, (12, 512, 512), (262144, 512, 1), 0), permute_291, out=buf216)
    del permute_291
    buf217 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf218 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf219 = buf184; del buf184  # reuse
    cpp_fused_clone_div_sqrt_sum_38(c_void_p(buf211.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()))
    buf220 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf219, (2304, 512), (1, 2304), 0), view_108, out=buf220)
    del view_108
    buf221 = reinterpret_tensor(buf216, (512, 768), (768, 1), 0); del buf216  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf219, (512, 2304), (2304, 1), 0), permute_298, out=buf221)
    del permute_298
    buf222 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf223 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf224 = buf205; del buf205  # reuse
    buf225 = buf204; del buf204  # reuse
    buf226 = buf203; del buf203  # reuse
    buf227 = buf206; del buf206  # reuse
    buf228 = reinterpret_tensor(buf215, (1, 512, 768), (393216, 768, 1), 0); del buf215  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_39(c_void_p(buf227.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(sub_49.data_ptr()), c_void_p(sqrt_18.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(convert_element_type_24.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf228.data_ptr()))
    del convert_element_type_24
    del primals_37
    del sqrt_18
    del sub_49
    buf229 = reinterpret_tensor(buf197, (512, 3072), (3072, 1), 0); del buf197  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf228, (512, 768), (768, 1), 0), permute_300, out=buf229)
    del permute_300
    buf230 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf228, (768, 512), (1, 768), 0), view_106, out=buf230)
    del view_106
    buf231 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf232 = reinterpret_tensor(buf229, (1, 512, 3072), (1572864, 3072, 1), 0); del buf229  # reuse
    cpp_fused_gelu_gelu_backward_sum_40(c_void_p(buf232.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(addmm_16.data_ptr()), c_void_p(buf231.data_ptr()))
    del addmm_16
    buf233 = reinterpret_tensor(buf228, (512, 768), (768, 1), 0); del buf228  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf232, (512, 3072), (3072, 1), 0), permute_304, out=buf233)
    del permute_304
    buf234 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf232, (3072, 512), (1, 3072), 0), view_104, out=buf234)
    del view_104
    buf235 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf236 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf237 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf238 = buf226; del buf226  # reuse
    buf239 = buf225; del buf225  # reuse
    buf240 = buf224; del buf224  # reuse
    buf241 = buf227; del buf227  # reuse
    buf242 = reinterpret_tensor(buf221, (1, 512, 768), (393216, 768, 1), 0); del buf221  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_41(c_void_p(buf241.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(sub_46.data_ptr()), c_void_p(sqrt_17.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(convert_element_type_23.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf242.data_ptr()))
    del convert_element_type_23
    del primals_35
    del sqrt_17
    del sub_46
    buf243 = buf233; del buf233  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf242, (512, 768), (768, 1), 0), permute_308, out=buf243)
    del permute_308
    buf244 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf242, (768, 512), (1, 768), 0), view_102, out=buf244)
    del view_102
    buf245 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_42(c_void_p(buf242.data_ptr()), c_void_p(buf245.data_ptr()))
    buf246 = reinterpret_tensor(buf242, (12, 512, 64), (32768, 64, 1), 0); del buf242  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_313, reinterpret_tensor(buf243, (12, 512, 64), (64, 768, 1), 0), out=buf246)
    del permute_313
    buf247 = reinterpret_tensor(buf214, (12, 512, 512), (262144, 512, 1), 0); del buf214  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf243, (12, 512, 64), (64, 768, 1), 0), permute_314, out=buf247)
    del permute_314
    buf248 = buf213; del buf213  # reuse
    buf249 = reinterpret_tensor(buf247, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf247  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_43(c_void_p(buf249.data_ptr()), c_void_p(convert_element_type_22.data_ptr()), c_void_p(alias_75.data_ptr()), c_void_p(buf248.data_ptr()))
    del alias_75
    del convert_element_type_22
    buf250 = reinterpret_tensor(buf243, (12, 64, 512), (32768, 512, 1), 0); del buf243  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_315, reinterpret_tensor(buf249, (12, 512, 512), (262144, 512, 1), 0), out=buf250)
    del permute_315
    buf251 = buf211; del buf211  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf249, (12, 512, 512), (262144, 512, 1), 0), permute_316, out=buf251)
    del permute_316
    buf252 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf253 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf254 = buf219; del buf219  # reuse
    cpp_fused_clone_div_sqrt_sum_44(c_void_p(buf246.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()))
    buf255 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf254, (2304, 512), (1, 2304), 0), view_90, out=buf255)
    del view_90
    buf256 = reinterpret_tensor(buf251, (512, 768), (768, 1), 0); del buf251  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf254, (512, 2304), (2304, 1), 0), permute_323, out=buf256)
    del permute_323
    buf257 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf258 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf259 = buf240; del buf240  # reuse
    buf260 = buf239; del buf239  # reuse
    buf261 = buf238; del buf238  # reuse
    buf262 = buf241; del buf241  # reuse
    buf263 = reinterpret_tensor(buf250, (1, 512, 768), (393216, 768, 1), 0); del buf250  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_45(c_void_p(buf262.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(sub_41.data_ptr()), c_void_p(sqrt_15.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(convert_element_type_20.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf263.data_ptr()))
    del convert_element_type_20
    del primals_31
    del sqrt_15
    del sub_41
    buf264 = reinterpret_tensor(buf232, (512, 3072), (3072, 1), 0); del buf232  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf263, (512, 768), (768, 1), 0), permute_325, out=buf264)
    del permute_325
    buf265 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf263, (768, 512), (1, 768), 0), view_88, out=buf265)
    del view_88
    buf266 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf267 = reinterpret_tensor(buf264, (1, 512, 3072), (1572864, 3072, 1), 0); del buf264  # reuse
    cpp_fused_gelu_gelu_backward_sum_46(c_void_p(buf267.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(addmm_13.data_ptr()), c_void_p(buf266.data_ptr()))
    del addmm_13
    buf268 = reinterpret_tensor(buf263, (512, 768), (768, 1), 0); del buf263  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf267, (512, 3072), (3072, 1), 0), permute_329, out=buf268)
    del permute_329
    buf269 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf267, (3072, 512), (1, 3072), 0), view_86, out=buf269)
    del view_86
    buf270 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf271 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf272 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf273 = buf261; del buf261  # reuse
    buf274 = buf260; del buf260  # reuse
    buf275 = buf259; del buf259  # reuse
    buf276 = buf262; del buf262  # reuse
    buf277 = reinterpret_tensor(buf256, (1, 512, 768), (393216, 768, 1), 0); del buf256  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_47(c_void_p(buf276.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(sub_38.data_ptr()), c_void_p(sqrt_14.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(convert_element_type_19.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf277.data_ptr()))
    del convert_element_type_19
    del primals_29
    del sqrt_14
    del sub_38
    buf278 = buf268; del buf268  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf277, (512, 768), (768, 1), 0), permute_333, out=buf278)
    del permute_333
    buf279 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf277, (768, 512), (1, 768), 0), view_84, out=buf279)
    del view_84
    buf280 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_48(c_void_p(buf277.data_ptr()), c_void_p(buf280.data_ptr()))
    buf281 = reinterpret_tensor(buf277, (12, 512, 64), (32768, 64, 1), 0); del buf277  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_338, reinterpret_tensor(buf278, (12, 512, 64), (64, 768, 1), 0), out=buf281)
    del permute_338
    buf282 = reinterpret_tensor(buf249, (12, 512, 512), (262144, 512, 1), 0); del buf249  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf278, (12, 512, 64), (64, 768, 1), 0), permute_339, out=buf282)
    del permute_339
    buf283 = buf248; del buf248  # reuse
    buf284 = reinterpret_tensor(buf282, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf282  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_49(c_void_p(buf284.data_ptr()), c_void_p(convert_element_type_18.data_ptr()), c_void_p(alias_80.data_ptr()), c_void_p(buf283.data_ptr()))
    del alias_80
    del convert_element_type_18
    buf285 = reinterpret_tensor(buf278, (12, 64, 512), (32768, 512, 1), 0); del buf278  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_340, reinterpret_tensor(buf284, (12, 512, 512), (262144, 512, 1), 0), out=buf285)
    del permute_340
    buf286 = buf246; del buf246  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf284, (12, 512, 512), (262144, 512, 1), 0), permute_341, out=buf286)
    del permute_341
    buf287 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf288 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf289 = buf254; del buf254  # reuse
    cpp_fused_clone_div_sqrt_sum_50(c_void_p(buf281.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()))
    buf290 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf289, (2304, 512), (1, 2304), 0), view_72, out=buf290)
    del view_72
    buf291 = reinterpret_tensor(buf286, (512, 768), (768, 1), 0); del buf286  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf289, (512, 2304), (2304, 1), 0), permute_348, out=buf291)
    del permute_348
    buf292 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf293 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf294 = buf275; del buf275  # reuse
    buf295 = buf274; del buf274  # reuse
    buf296 = buf273; del buf273  # reuse
    buf297 = buf276; del buf276  # reuse
    buf298 = reinterpret_tensor(buf285, (1, 512, 768), (393216, 768, 1), 0); del buf285  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_51(c_void_p(buf297.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(sub_33.data_ptr()), c_void_p(sqrt_12.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(convert_element_type_16.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf298.data_ptr()))
    del convert_element_type_16
    del primals_25
    del sqrt_12
    del sub_33
    buf299 = reinterpret_tensor(buf267, (512, 3072), (3072, 1), 0); del buf267  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf298, (512, 768), (768, 1), 0), permute_350, out=buf299)
    del permute_350
    buf300 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf298, (768, 512), (1, 768), 0), view_70, out=buf300)
    del view_70
    buf301 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf302 = reinterpret_tensor(buf299, (1, 512, 3072), (1572864, 3072, 1), 0); del buf299  # reuse
    cpp_fused_gelu_gelu_backward_sum_52(c_void_p(buf302.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(buf301.data_ptr()))
    del addmm_10
    buf303 = reinterpret_tensor(buf298, (512, 768), (768, 1), 0); del buf298  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf302, (512, 3072), (3072, 1), 0), permute_354, out=buf303)
    del permute_354
    buf304 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf302, (3072, 512), (1, 3072), 0), view_68, out=buf304)
    del view_68
    buf305 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf306 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf307 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf308 = buf296; del buf296  # reuse
    buf309 = buf295; del buf295  # reuse
    buf310 = buf294; del buf294  # reuse
    buf311 = buf297; del buf297  # reuse
    buf312 = reinterpret_tensor(buf291, (1, 512, 768), (393216, 768, 1), 0); del buf291  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_53(c_void_p(buf311.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(sub_30.data_ptr()), c_void_p(sqrt_11.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(convert_element_type_15.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf312.data_ptr()))
    del convert_element_type_15
    del primals_23
    del sqrt_11
    del sub_30
    buf313 = buf303; del buf303  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf312, (512, 768), (768, 1), 0), permute_358, out=buf313)
    del permute_358
    buf314 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf312, (768, 512), (1, 768), 0), view_66, out=buf314)
    del view_66
    buf315 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_54(c_void_p(buf312.data_ptr()), c_void_p(buf315.data_ptr()))
    buf316 = reinterpret_tensor(buf312, (12, 512, 64), (32768, 64, 1), 0); del buf312  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_363, reinterpret_tensor(buf313, (12, 512, 64), (64, 768, 1), 0), out=buf316)
    del permute_363
    buf317 = reinterpret_tensor(buf284, (12, 512, 512), (262144, 512, 1), 0); del buf284  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf313, (12, 512, 64), (64, 768, 1), 0), permute_364, out=buf317)
    del permute_364
    buf318 = buf283; del buf283  # reuse
    buf319 = reinterpret_tensor(buf317, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf317  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_55(c_void_p(buf319.data_ptr()), c_void_p(convert_element_type_14.data_ptr()), c_void_p(alias_85.data_ptr()), c_void_p(buf318.data_ptr()))
    del alias_85
    del convert_element_type_14
    buf320 = reinterpret_tensor(buf313, (12, 64, 512), (32768, 512, 1), 0); del buf313  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_365, reinterpret_tensor(buf319, (12, 512, 512), (262144, 512, 1), 0), out=buf320)
    del permute_365
    buf321 = buf281; del buf281  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf319, (12, 512, 512), (262144, 512, 1), 0), permute_366, out=buf321)
    del permute_366
    buf322 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf323 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf324 = buf289; del buf289  # reuse
    cpp_fused_clone_div_sqrt_sum_56(c_void_p(buf316.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()))
    buf325 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf324, (2304, 512), (1, 2304), 0), view_54, out=buf325)
    del view_54
    buf326 = reinterpret_tensor(buf321, (512, 768), (768, 1), 0); del buf321  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf324, (512, 2304), (2304, 1), 0), permute_373, out=buf326)
    del permute_373
    buf327 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf328 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf329 = buf310; del buf310  # reuse
    buf330 = buf309; del buf309  # reuse
    buf331 = buf308; del buf308  # reuse
    buf332 = buf311; del buf311  # reuse
    buf333 = reinterpret_tensor(buf320, (1, 512, 768), (393216, 768, 1), 0); del buf320  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_57(c_void_p(buf332.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(sub_25.data_ptr()), c_void_p(sqrt_9.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(convert_element_type_12.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf333.data_ptr()))
    del convert_element_type_12
    del primals_19
    del sqrt_9
    del sub_25
    buf334 = reinterpret_tensor(buf302, (512, 3072), (3072, 1), 0); del buf302  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf333, (512, 768), (768, 1), 0), permute_375, out=buf334)
    del permute_375
    buf335 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf333, (768, 512), (1, 768), 0), view_52, out=buf335)
    del view_52
    buf336 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf337 = reinterpret_tensor(buf334, (1, 512, 3072), (1572864, 3072, 1), 0); del buf334  # reuse
    cpp_fused_gelu_gelu_backward_sum_58(c_void_p(buf337.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(addmm_7.data_ptr()), c_void_p(buf336.data_ptr()))
    del addmm_7
    buf338 = reinterpret_tensor(buf333, (512, 768), (768, 1), 0); del buf333  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf337, (512, 3072), (3072, 1), 0), permute_379, out=buf338)
    del permute_379
    buf339 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf337, (3072, 512), (1, 3072), 0), view_50, out=buf339)
    del view_50
    buf340 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf341 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf342 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf343 = buf331; del buf331  # reuse
    buf344 = buf330; del buf330  # reuse
    buf345 = buf329; del buf329  # reuse
    buf346 = buf332; del buf332  # reuse
    buf347 = reinterpret_tensor(buf326, (1, 512, 768), (393216, 768, 1), 0); del buf326  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_59(c_void_p(buf346.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(sub_22.data_ptr()), c_void_p(sqrt_8.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(convert_element_type_11.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf347.data_ptr()))
    del convert_element_type_11
    del primals_17
    del sqrt_8
    del sub_22
    buf348 = buf338; del buf338  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf347, (512, 768), (768, 1), 0), permute_383, out=buf348)
    del permute_383
    buf349 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf347, (768, 512), (1, 768), 0), view_48, out=buf349)
    del view_48
    buf350 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_60(c_void_p(buf347.data_ptr()), c_void_p(buf350.data_ptr()))
    buf351 = reinterpret_tensor(buf347, (12, 512, 64), (32768, 64, 1), 0); del buf347  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_388, reinterpret_tensor(buf348, (12, 512, 64), (64, 768, 1), 0), out=buf351)
    del permute_388
    buf352 = reinterpret_tensor(buf319, (12, 512, 512), (262144, 512, 1), 0); del buf319  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf348, (12, 512, 64), (64, 768, 1), 0), permute_389, out=buf352)
    del permute_389
    buf353 = buf318; del buf318  # reuse
    buf354 = reinterpret_tensor(buf352, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf352  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_61(c_void_p(buf354.data_ptr()), c_void_p(convert_element_type_10.data_ptr()), c_void_p(alias_90.data_ptr()), c_void_p(buf353.data_ptr()))
    del alias_90
    del convert_element_type_10
    buf355 = reinterpret_tensor(buf348, (12, 64, 512), (32768, 512, 1), 0); del buf348  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_390, reinterpret_tensor(buf354, (12, 512, 512), (262144, 512, 1), 0), out=buf355)
    del permute_390
    buf356 = buf316; del buf316  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf354, (12, 512, 512), (262144, 512, 1), 0), permute_391, out=buf356)
    del permute_391
    buf357 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf358 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf359 = buf324; del buf324  # reuse
    cpp_fused_clone_div_sqrt_sum_62(c_void_p(buf351.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()))
    buf360 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf359, (2304, 512), (1, 2304), 0), view_36, out=buf360)
    del view_36
    buf361 = reinterpret_tensor(buf356, (512, 768), (768, 1), 0); del buf356  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf359, (512, 2304), (2304, 1), 0), permute_398, out=buf361)
    del permute_398
    buf362 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf363 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf364 = buf345; del buf345  # reuse
    buf365 = buf344; del buf344  # reuse
    buf366 = buf343; del buf343  # reuse
    buf367 = buf346; del buf346  # reuse
    buf368 = reinterpret_tensor(buf355, (1, 512, 768), (393216, 768, 1), 0); del buf355  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_63(c_void_p(buf367.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(sub_17.data_ptr()), c_void_p(sqrt_6.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(convert_element_type_8.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf368.data_ptr()))
    del convert_element_type_8
    del primals_13
    del sqrt_6
    del sub_17
    buf369 = reinterpret_tensor(buf337, (512, 3072), (3072, 1), 0); del buf337  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf368, (512, 768), (768, 1), 0), permute_400, out=buf369)
    del permute_400
    buf370 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf368, (768, 512), (1, 768), 0), view_34, out=buf370)
    del view_34
    buf371 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf372 = reinterpret_tensor(buf369, (1, 512, 3072), (1572864, 3072, 1), 0); del buf369  # reuse
    cpp_fused_gelu_gelu_backward_sum_64(c_void_p(buf372.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(addmm_4.data_ptr()), c_void_p(buf371.data_ptr()))
    del addmm_4
    buf373 = reinterpret_tensor(buf368, (512, 768), (768, 1), 0); del buf368  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf372, (512, 3072), (3072, 1), 0), permute_404, out=buf373)
    del permute_404
    buf374 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf372, (3072, 512), (1, 3072), 0), view_32, out=buf374)
    del view_32
    buf375 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf376 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf377 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf378 = buf366; del buf366  # reuse
    buf379 = buf365; del buf365  # reuse
    buf380 = buf364; del buf364  # reuse
    buf381 = buf367; del buf367  # reuse
    buf382 = reinterpret_tensor(buf361, (1, 512, 768), (393216, 768, 1), 0); del buf361  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_65(c_void_p(buf381.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(sub_14.data_ptr()), c_void_p(sqrt_5.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(convert_element_type_7.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf382.data_ptr()))
    del convert_element_type_7
    del primals_11
    del sqrt_5
    del sub_14
    buf383 = buf373; del buf373  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf382, (512, 768), (768, 1), 0), permute_408, out=buf383)
    del permute_408
    buf384 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf382, (768, 512), (1, 768), 0), view_30, out=buf384)
    del view_30
    buf385 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_66(c_void_p(buf382.data_ptr()), c_void_p(buf385.data_ptr()))
    buf386 = reinterpret_tensor(buf382, (12, 512, 64), (32768, 64, 1), 0); del buf382  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_413, reinterpret_tensor(buf383, (12, 512, 64), (64, 768, 1), 0), out=buf386)
    del permute_413
    buf387 = reinterpret_tensor(buf354, (12, 512, 512), (262144, 512, 1), 0); del buf354  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf383, (12, 512, 64), (64, 768, 1), 0), permute_414, out=buf387)
    del permute_414
    buf388 = buf353; del buf353  # reuse
    buf389 = reinterpret_tensor(buf387, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf387  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_67(c_void_p(buf389.data_ptr()), c_void_p(convert_element_type_6.data_ptr()), c_void_p(alias_95.data_ptr()), c_void_p(buf388.data_ptr()))
    del alias_95
    del convert_element_type_6
    buf390 = reinterpret_tensor(buf383, (12, 64, 512), (32768, 512, 1), 0); del buf383  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_415, reinterpret_tensor(buf389, (12, 512, 512), (262144, 512, 1), 0), out=buf390)
    del permute_415
    buf391 = buf351; del buf351  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf389, (12, 512, 512), (262144, 512, 1), 0), permute_416, out=buf391)
    del permute_416
    buf392 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf393 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf394 = buf359; del buf359  # reuse
    cpp_fused_clone_div_sqrt_sum_68(c_void_p(buf386.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()))
    buf395 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf394, (2304, 512), (1, 2304), 0), view_18, out=buf395)
    del view_18
    buf396 = reinterpret_tensor(buf391, (512, 768), (768, 1), 0); del buf391  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf394, (512, 2304), (2304, 1), 0), permute_423, out=buf396)
    del permute_423
    buf397 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf398 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf399 = buf380; del buf380  # reuse
    buf400 = buf379; del buf379  # reuse
    buf401 = buf378; del buf378  # reuse
    buf402 = buf381; del buf381  # reuse
    buf403 = reinterpret_tensor(buf390, (1, 512, 768), (393216, 768, 1), 0); del buf390  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_69(c_void_p(buf402.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(sub_9.data_ptr()), c_void_p(sqrt_3.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(convert_element_type_4.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf403.data_ptr()))
    del convert_element_type_4
    del primals_7
    del sqrt_3
    del sub_9
    buf404 = reinterpret_tensor(buf372, (512, 3072), (3072, 1), 0); del buf372  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf403, (512, 768), (768, 1), 0), permute_425, out=buf404)
    del permute_425
    buf405 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf403, (768, 512), (1, 768), 0), view_16, out=buf405)
    del view_16
    buf406 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf407 = reinterpret_tensor(buf404, (1, 512, 3072), (1572864, 3072, 1), 0); del buf404  # reuse
    cpp_fused_gelu_gelu_backward_sum_70(c_void_p(buf407.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(addmm_1.data_ptr()), c_void_p(buf406.data_ptr()))
    del addmm_1
    buf408 = reinterpret_tensor(buf403, (512, 768), (768, 1), 0); del buf403  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf407, (512, 3072), (3072, 1), 0), permute_429, out=buf408)
    del permute_429
    buf409 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf407, (3072, 512), (1, 3072), 0), view_14, out=buf409)
    del view_14
    buf410 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf411 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf412 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf413 = buf401; del buf401  # reuse
    buf414 = buf400; del buf400  # reuse
    buf415 = buf399; del buf399  # reuse
    buf416 = buf402; del buf402  # reuse
    buf417 = reinterpret_tensor(buf396, (1, 512, 768), (393216, 768, 1), 0); del buf396  # reuse
    cpp_fused_add_div_masked_fill_mul_neg_pow_sum_71(c_void_p(buf416.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(sub_6.data_ptr()), c_void_p(sqrt_2.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(convert_element_type_3.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf417.data_ptr()))
    del buf407
    del convert_element_type_3
    del primals_5
    del sqrt_2
    del sub_6
    buf418 = buf408; del buf408  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf417, (512, 768), (768, 1), 0), permute_433, out=buf418)
    del permute_433
    buf419 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf417, (768, 512), (1, 768), 0), view_12, out=buf419)
    del view_12
    buf420 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_72(c_void_p(buf417.data_ptr()), c_void_p(buf420.data_ptr()))
    buf421 = reinterpret_tensor(buf417, (12, 512, 64), (32768, 64, 1), 0); del buf417  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_438, reinterpret_tensor(buf418, (12, 512, 64), (64, 768, 1), 0), out=buf421)
    del permute_438
    buf422 = reinterpret_tensor(buf389, (12, 512, 512), (262144, 512, 1), 0); del buf389  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf418, (12, 512, 64), (64, 768, 1), 0), permute_439, out=buf422)
    del permute_439
    buf423 = buf388; del buf388  # reuse
    buf424 = reinterpret_tensor(buf422, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf422  # reuse
    cpp_fused__softmax_backward_data_masked_fill_mul_73(c_void_p(buf424.data_ptr()), c_void_p(convert_element_type_2.data_ptr()), c_void_p(alias_100.data_ptr()), c_void_p(buf423.data_ptr()))
    del alias_100
    del buf423
    del convert_element_type_2
    buf425 = reinterpret_tensor(buf418, (12, 64, 512), (32768, 512, 1), 0); del buf418  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_440, reinterpret_tensor(buf424, (12, 512, 512), (262144, 512, 1), 0), out=buf425)
    del permute_440
    buf426 = buf386; del buf386  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf424, (12, 512, 512), (262144, 512, 1), 0), permute_441, out=buf426)
    del buf424
    del permute_441
    buf427 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf428 = empty_strided((1, 12, 1, 64), (768, 64, 768, 1), device='cpu', dtype=torch.float32)
    buf429 = buf394; del buf394  # reuse
    cpp_fused_clone_div_sqrt_sum_74(c_void_p(buf421.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()))
    buf430 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf429, (2304, 512), (1, 2304), 0), view, out=buf430)
    del view
    buf431 = reinterpret_tensor(buf426, (512, 768), (768, 1), 0); del buf426  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf429, (512, 2304), (2304, 1), 0), permute_448, out=buf431)
    del buf429
    del permute_448
    buf432 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf433 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf434 = reinterpret_tensor(buf425, (1, 512, 768), (393216, 768, 1), 0); del buf425  # reuse
    buf435 = buf415; del buf415  # reuse
    buf436 = buf414; del buf414  # reuse
    buf437 = buf413; del buf413  # reuse
    buf438 = buf434; del buf434  # reuse
    buf440 = reinterpret_tensor(buf421, (1, 512, 768), (393216, 768, 1), 0); del buf421  # reuse
    buf444 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf439 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_embedding_dense_backward_masked_fill_mul_neg_pow_sum_75(c_void_p(buf438.data_ptr()), c_void_p(convert_element_type.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(sub.data_ptr()), c_void_p(sqrt.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(slice_1.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf439.data_ptr()))
    del buf416
    del buf431
    del buf435
    del buf436
    del buf437
    del buf438
    del convert_element_type
    del primals_1
    del sqrt
    del sub
    aten.index_put_(buf439, [slice_1], buf440, True)
    del buf440
    del slice_1
    buf443 = empty((50265, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_76(c_void_p(buf443.data_ptr()))
    aten.index_put_(buf443, [primals_164], buf444, True)
    del buf444
    del primals_164
    return (reinterpret_tensor(buf433, (768, ), (1, ), 0), reinterpret_tensor(buf432, (768, ), (1, ), 0), reinterpret_tensor(buf428, (768, ), (1, ), 0), reinterpret_tensor(buf427, (768, ), (1, ), 0), reinterpret_tensor(buf412, (768, ), (1, ), 0), reinterpret_tensor(buf411, (768, ), (1, ), 0), reinterpret_tensor(buf398, (768, ), (1, ), 0), reinterpret_tensor(buf397, (768, ), (1, ), 0), reinterpret_tensor(buf393, (768, ), (1, ), 0), reinterpret_tensor(buf392, (768, ), (1, ), 0), reinterpret_tensor(buf377, (768, ), (1, ), 0), reinterpret_tensor(buf376, (768, ), (1, ), 0), reinterpret_tensor(buf363, (768, ), (1, ), 0), reinterpret_tensor(buf362, (768, ), (1, ), 0), reinterpret_tensor(buf358, (768, ), (1, ), 0), reinterpret_tensor(buf357, (768, ), (1, ), 0), reinterpret_tensor(buf342, (768, ), (1, ), 0), reinterpret_tensor(buf341, (768, ), (1, ), 0), reinterpret_tensor(buf328, (768, ), (1, ), 0), reinterpret_tensor(buf327, (768, ), (1, ), 0), reinterpret_tensor(buf323, (768, ), (1, ), 0), reinterpret_tensor(buf322, (768, ), (1, ), 0), reinterpret_tensor(buf307, (768, ), (1, ), 0), reinterpret_tensor(buf306, (768, ), (1, ), 0), reinterpret_tensor(buf293, (768, ), (1, ), 0), reinterpret_tensor(buf292, (768, ), (1, ), 0), reinterpret_tensor(buf288, (768, ), (1, ), 0), reinterpret_tensor(buf287, (768, ), (1, ), 0), reinterpret_tensor(buf272, (768, ), (1, ), 0), reinterpret_tensor(buf271, (768, ), (1, ), 0), reinterpret_tensor(buf258, (768, ), (1, ), 0), reinterpret_tensor(buf257, (768, ), (1, ), 0), reinterpret_tensor(buf253, (768, ), (1, ), 0), reinterpret_tensor(buf252, (768, ), (1, ), 0), reinterpret_tensor(buf237, (768, ), (1, ), 0), reinterpret_tensor(buf236, (768, ), (1, ), 0), reinterpret_tensor(buf223, (768, ), (1, ), 0), reinterpret_tensor(buf222, (768, ), (1, ), 0), reinterpret_tensor(buf218, (768, ), (1, ), 0), reinterpret_tensor(buf217, (768, ), (1, ), 0), reinterpret_tensor(buf202, (768, ), (1, ), 0), reinterpret_tensor(buf201, (768, ), (1, ), 0), reinterpret_tensor(buf188, (768, ), (1, ), 0), reinterpret_tensor(buf187, (768, ), (1, ), 0), reinterpret_tensor(buf183, (768, ), (1, ), 0), reinterpret_tensor(buf182, (768, ), (1, ), 0), reinterpret_tensor(buf167, (768, ), (1, ), 0), reinterpret_tensor(buf166, (768, ), (1, ), 0), reinterpret_tensor(buf153, (768, ), (1, ), 0), reinterpret_tensor(buf152, (768, ), (1, ), 0), reinterpret_tensor(buf148, (768, ), (1, ), 0), reinterpret_tensor(buf147, (768, ), (1, ), 0), reinterpret_tensor(buf132, (768, ), (1, ), 0), reinterpret_tensor(buf131, (768, ), (1, ), 0), reinterpret_tensor(buf118, (768, ), (1, ), 0), reinterpret_tensor(buf117, (768, ), (1, ), 0), reinterpret_tensor(buf113, (768, ), (1, ), 0), reinterpret_tensor(buf112, (768, ), (1, ), 0), reinterpret_tensor(buf97, (768, ), (1, ), 0), reinterpret_tensor(buf96, (768, ), (1, ), 0), reinterpret_tensor(buf83, (768, ), (1, ), 0), reinterpret_tensor(buf82, (768, ), (1, ), 0), reinterpret_tensor(buf78, (768, ), (1, ), 0), reinterpret_tensor(buf77, (768, ), (1, ), 0), reinterpret_tensor(buf62, (768, ), (1, ), 0), reinterpret_tensor(buf61, (768, ), (1, ), 0), reinterpret_tensor(buf48, (768, ), (1, ), 0), reinterpret_tensor(buf47, (768, ), (1, ), 0), reinterpret_tensor(buf43, (768, ), (1, ), 0), reinterpret_tensor(buf42, (768, ), (1, ), 0), reinterpret_tensor(buf27, (768, ), (1, ), 0), reinterpret_tensor(buf26, (768, ), (1, ), 0), reinterpret_tensor(buf13, (768, ), (1, ), 0), reinterpret_tensor(buf12, (768, ), (1, ), 0), buf443, buf439, reinterpret_tensor(buf430, (2304, 768), (768, 1), 0), reinterpret_tensor(buf419, (768, 768), (768, 1), 0), reinterpret_tensor(buf420, (768, ), (1, ), 0), reinterpret_tensor(buf409, (3072, 768), (768, 1), 0), reinterpret_tensor(buf410, (3072, ), (1, ), 0), reinterpret_tensor(buf405, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf406, (768, ), (1, ), 0), reinterpret_tensor(buf395, (2304, 768), (768, 1), 0), reinterpret_tensor(buf384, (768, 768), (768, 1), 0), reinterpret_tensor(buf385, (768, ), (1, ), 0), reinterpret_tensor(buf374, (3072, 768), (768, 1), 0), reinterpret_tensor(buf375, (3072, ), (1, ), 0), reinterpret_tensor(buf370, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf371, (768, ), (1, ), 0), reinterpret_tensor(buf360, (2304, 768), (768, 1), 0), reinterpret_tensor(buf349, (768, 768), (768, 1), 0), reinterpret_tensor(buf350, (768, ), (1, ), 0), reinterpret_tensor(buf339, (3072, 768), (768, 1), 0), reinterpret_tensor(buf340, (3072, ), (1, ), 0), reinterpret_tensor(buf335, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf336, (768, ), (1, ), 0), reinterpret_tensor(buf325, (2304, 768), (768, 1), 0), reinterpret_tensor(buf314, (768, 768), (768, 1), 0), reinterpret_tensor(buf315, (768, ), (1, ), 0), reinterpret_tensor(buf304, (3072, 768), (768, 1), 0), reinterpret_tensor(buf305, (3072, ), (1, ), 0), reinterpret_tensor(buf300, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf301, (768, ), (1, ), 0), reinterpret_tensor(buf290, (2304, 768), (768, 1), 0), reinterpret_tensor(buf279, (768, 768), (768, 1), 0), reinterpret_tensor(buf280, (768, ), (1, ), 0), reinterpret_tensor(buf269, (3072, 768), (768, 1), 0), reinterpret_tensor(buf270, (3072, ), (1, ), 0), reinterpret_tensor(buf265, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf266, (768, ), (1, ), 0), reinterpret_tensor(buf255, (2304, 768), (768, 1), 0), reinterpret_tensor(buf244, (768, 768), (768, 1), 0), reinterpret_tensor(buf245, (768, ), (1, ), 0), reinterpret_tensor(buf234, (3072, 768), (768, 1), 0), reinterpret_tensor(buf235, (3072, ), (1, ), 0), reinterpret_tensor(buf230, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf231, (768, ), (1, ), 0), reinterpret_tensor(buf220, (2304, 768), (768, 1), 0), reinterpret_tensor(buf209, (768, 768), (768, 1), 0), reinterpret_tensor(buf210, (768, ), (1, ), 0), reinterpret_tensor(buf199, (3072, 768), (768, 1), 0), reinterpret_tensor(buf200, (3072, ), (1, ), 0), reinterpret_tensor(buf195, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf196, (768, ), (1, ), 0), reinterpret_tensor(buf185, (2304, 768), (768, 1), 0), reinterpret_tensor(buf174, (768, 768), (768, 1), 0), reinterpret_tensor(buf175, (768, ), (1, ), 0), reinterpret_tensor(buf164, (3072, 768), (768, 1), 0), reinterpret_tensor(buf165, (3072, ), (1, ), 0), reinterpret_tensor(buf160, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf161, (768, ), (1, ), 0), reinterpret_tensor(buf150, (2304, 768), (768, 1), 0), reinterpret_tensor(buf139, (768, 768), (768, 1), 0), reinterpret_tensor(buf140, (768, ), (1, ), 0), reinterpret_tensor(buf129, (3072, 768), (768, 1), 0), reinterpret_tensor(buf130, (3072, ), (1, ), 0), reinterpret_tensor(buf125, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf126, (768, ), (1, ), 0), reinterpret_tensor(buf115, (2304, 768), (768, 1), 0), reinterpret_tensor(buf104, (768, 768), (768, 1), 0), reinterpret_tensor(buf105, (768, ), (1, ), 0), reinterpret_tensor(buf94, (3072, 768), (768, 1), 0), reinterpret_tensor(buf95, (3072, ), (1, ), 0), reinterpret_tensor(buf90, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf91, (768, ), (1, ), 0), reinterpret_tensor(buf80, (2304, 768), (768, 1), 0), reinterpret_tensor(buf69, (768, 768), (768, 1), 0), reinterpret_tensor(buf70, (768, ), (1, ), 0), reinterpret_tensor(buf59, (3072, 768), (768, 1), 0), reinterpret_tensor(buf60, (3072, ), (1, ), 0), reinterpret_tensor(buf55, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf56, (768, ), (1, ), 0), reinterpret_tensor(buf45, (2304, 768), (768, 1), 0), reinterpret_tensor(buf34, (768, 768), (768, 1), 0), reinterpret_tensor(buf35, (768, ), (1, ), 0), reinterpret_tensor(buf24, (3072, 768), (768, 1), 0), reinterpret_tensor(buf25, (3072, ), (1, ), 0), reinterpret_tensor(buf20, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf21, (768, ), (1, ), 0), reinterpret_tensor(buf10, (2, 768), (768, 1), 0), reinterpret_tensor(buf11, (2, ), (1, ), 0), None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    slice_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    sub = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    convert_element_type = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    view = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_2 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_12 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_3 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_6 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_2 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_14 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_1 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_16 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_4 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_9 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_3 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_18 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_6 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_30 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_7 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_14 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_5 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_32 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_4 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_34 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_8 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_17 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_6 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_36 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_10 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_48 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_11 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_22 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_8 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_50 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_7 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_52 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_12 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_25 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_9 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_54 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_14 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_66 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_15 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_30 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_11 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_68 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_70 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_16 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_33 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_12 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_72 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_18 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_84 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_19 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_38 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_14 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_86 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_13 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_88 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_20 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_41 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_15 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_90 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_22 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_102 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_23 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_46 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_17 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_104 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_16 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_106 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_24 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_49 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_18 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_108 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_26 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_120 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_27 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_54 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_20 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_122 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_19 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_124 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_28 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_57 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_21 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_126 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_30 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_138 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_31 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_62 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_23 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_140 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_142 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_32 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_65 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_24 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_144 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_34 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_156 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_35 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_70 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_26 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_158 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_25 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_160 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_36 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_73 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_27 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_162 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_38 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_174 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_39 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_78 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_29 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_176 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_28 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_178 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_40 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_81 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_30 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_180 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_42 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_192 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_43 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_86 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_32 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_194 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_31 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_196 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_44 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_89 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_33 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_198 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_46 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    view_210 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    convert_element_type_47 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_94 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_35 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_212 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_214 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    convert_element_type_48 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    sub_97 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    sqrt_36 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    view_216 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    sub_100 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    ne = rand_strided((1, ), (1, ), device='cpu', dtype=torch.bool)
    sub_102 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    ne_3 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.bool)
    ne_6 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.bool)
    where_65 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.int64)
    ne_8 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.bool)
    where_67 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.int64)
    permute_146 = rand_strided((2, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_150 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_154 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_158 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_163 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_164 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_45 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_165 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_166 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_173 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_175 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_179 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_183 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_188 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_189 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_50 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_190 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_191 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_198 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_200 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_204 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_208 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_213 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_214 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_55 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_215 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_216 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_223 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_225 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_229 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_233 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_238 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_239 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_60 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_240 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_241 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_248 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_250 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_254 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_258 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_263 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_264 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_65 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_265 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_266 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_273 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_275 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_279 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_283 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_288 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_289 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_70 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_290 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_291 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_298 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_300 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_304 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_308 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_313 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_314 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_75 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_315 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_316 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_323 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_325 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_329 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_333 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_338 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_339 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_80 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_340 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_341 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_348 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_350 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_354 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_358 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_363 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_364 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_85 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_365 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_366 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_373 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_375 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_379 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_383 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_388 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_389 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_90 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_390 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_391 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_398 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_400 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_404 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_408 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_413 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_414 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_95 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_415 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_416 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_423 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_425 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_429 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_433 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_438 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_439 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_100 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_440 = rand_strided((12, 64, 512), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_441 = rand_strided((12, 512, 64), (192, 2304, 1), device='cpu', dtype=torch.float32)
    permute_448 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    tangents_3 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_5, primals_7, primals_11, primals_13, primals_17, primals_19, primals_23, primals_25, primals_29, primals_31, primals_35, primals_37, primals_41, primals_43, primals_47, primals_49, primals_53, primals_55, primals_59, primals_61, primals_65, primals_67, primals_71, primals_73, primals_164, slice_1, sub, sqrt, convert_element_type, view, convert_element_type_2, view_12, convert_element_type_3, sub_6, sqrt_2, view_14, addmm_1, view_16, convert_element_type_4, sub_9, sqrt_3, view_18, convert_element_type_6, view_30, convert_element_type_7, sub_14, sqrt_5, view_32, addmm_4, view_34, convert_element_type_8, sub_17, sqrt_6, view_36, convert_element_type_10, view_48, convert_element_type_11, sub_22, sqrt_8, view_50, addmm_7, view_52, convert_element_type_12, sub_25, sqrt_9, view_54, convert_element_type_14, view_66, convert_element_type_15, sub_30, sqrt_11, view_68, addmm_10, view_70, convert_element_type_16, sub_33, sqrt_12, view_72, convert_element_type_18, view_84, convert_element_type_19, sub_38, sqrt_14, view_86, addmm_13, view_88, convert_element_type_20, sub_41, sqrt_15, view_90, convert_element_type_22, view_102, convert_element_type_23, sub_46, sqrt_17, view_104, addmm_16, view_106, convert_element_type_24, sub_49, sqrt_18, view_108, convert_element_type_26, view_120, convert_element_type_27, sub_54, sqrt_20, view_122, addmm_19, view_124, convert_element_type_28, sub_57, sqrt_21, view_126, convert_element_type_30, view_138, convert_element_type_31, sub_62, sqrt_23, view_140, addmm_22, view_142, convert_element_type_32, sub_65, sqrt_24, view_144, convert_element_type_34, view_156, convert_element_type_35, sub_70, sqrt_26, view_158, addmm_25, view_160, convert_element_type_36, sub_73, sqrt_27, view_162, convert_element_type_38, view_174, convert_element_type_39, sub_78, sqrt_29, view_176, addmm_28, view_178, convert_element_type_40, sub_81, sqrt_30, view_180, convert_element_type_42, view_192, convert_element_type_43, sub_86, sqrt_32, view_194, addmm_31, view_196, convert_element_type_44, sub_89, sqrt_33, view_198, convert_element_type_46, view_210, convert_element_type_47, sub_94, sqrt_35, view_212, addmm_34, view_214, convert_element_type_48, sub_97, sqrt_36, view_216, sub_100, ne, sub_102, ne_3, ne_6, where_65, ne_8, where_67, permute_146, permute_150, permute_154, permute_158, permute_163, permute_164, alias_45, permute_165, permute_166, permute_173, permute_175, permute_179, permute_183, permute_188, permute_189, alias_50, permute_190, permute_191, permute_198, permute_200, permute_204, permute_208, permute_213, permute_214, alias_55, permute_215, permute_216, permute_223, permute_225, permute_229, permute_233, permute_238, permute_239, alias_60, permute_240, permute_241, permute_248, permute_250, permute_254, permute_258, permute_263, permute_264, alias_65, permute_265, permute_266, permute_273, permute_275, permute_279, permute_283, permute_288, permute_289, alias_70, permute_290, permute_291, permute_298, permute_300, permute_304, permute_308, permute_313, permute_314, alias_75, permute_315, permute_316, permute_323, permute_325, permute_329, permute_333, permute_338, permute_339, alias_80, permute_340, permute_341, permute_348, permute_350, permute_354, permute_358, permute_363, permute_364, alias_85, permute_365, permute_366, permute_373, permute_375, permute_379, permute_383, permute_388, permute_389, alias_90, permute_390, permute_391, permute_398, permute_400, permute_404, permute_408, permute_413, permute_414, alias_95, permute_415, permute_416, permute_423, permute_425, permute_429, permute_433, permute_438, permute_439, alias_100, permute_440, permute_441, permute_448, tangents_1, tangents_2, tangents_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DebertaForQuestionAnswering', benchmark_compiled_module)
