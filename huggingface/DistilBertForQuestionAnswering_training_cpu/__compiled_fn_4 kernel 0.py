
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                       const bool* in_ptr2,
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
    {
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (2L*x1))];
                    tmp_acc0 = tmp_acc0 + tmp0;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                    auto tmp1 = in_ptr2[static_cast<long>(x1 + (768L*x0))];
                    auto tmp6 = in_ptr3[static_cast<long>(x1)];
                    auto tmp8 = in_ptr4[static_cast<long>(x1 + (768L*x0))];
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr5[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                auto tmp2 = in_ptr2[static_cast<long>(x1 + (768L*x0))];
                auto tmp7 = in_ptr3[static_cast<long>(x1)];
                auto tmp11 = out_ptr1[static_cast<long>(x0)];
                auto tmp13 = in_ptr4[static_cast<long>(x1 + (768L*x0))];
                auto tmp14 = out_ptr2[static_cast<long>(x0)];
                auto tmp3 = c10::convert<float>(tmp2);
                auto tmp4 = static_cast<float>(1.1111111111111112);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                auto tmp9 = static_cast<float>(768.0);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp12 = decltype(tmp10)(tmp10 - tmp11);
                auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                auto tmp16 = decltype(tmp12)(tmp12 - tmp15);
                auto tmp17 = decltype(tmp0)(tmp0 * tmp16);
                out_ptr3[static_cast<long>(x1 + (768L*x0))] = tmp17;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr1[static_cast<long>(x0 + (768L*x1))];
                    auto tmp1 = in_ptr2[static_cast<long>(x0 + (768L*x1))];
                    auto tmp6 = in_ptr4[static_cast<long>(x0 + (768L*x1))];
                    auto tmp2 = c10::convert<float>(tmp1);
                    auto tmp3 = static_cast<float>(1.1111111111111112);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    tmp_acc0 = tmp_acc0 + tmp7;
                    tmp_acc1 = tmp_acc1 + tmp5;
                }
                out_ptr4[static_cast<long>(x0)] = tmp_acc0;
                out_ptr5[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_gelu_gelu_backward_sum_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_5 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        }
    }
}
''')


cpp_fused_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(x1)];
                    auto tmp1 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp2 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp7 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = c10::convert<float>(tmp2);
                    auto tmp4 = static_cast<float>(1.1111111111111112);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = tmp0 ? tmp12 : tmp11;
                    in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp13;
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_9 = async_compile.cpp('''
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
    {
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
''')


cpp_fused_sum_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_12 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr7[static_cast<long>(x0)];
            auto tmp2 = c10::convert<float>(tmp1);
            auto tmp3 = static_cast<float>(1.1111111111111112);
            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            out_ptr3[static_cast<long>(x0)] = tmp5;
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_14 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        }
    }
}
''')


cpp_fused_view_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(x1)];
                    auto tmp1 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp2 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp7 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = c10::convert<float>(tmp2);
                    auto tmp4 = static_cast<float>(1.1111111111111112);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = tmp0 ? tmp12 : tmp11;
                    in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp13;
                }
            }
        }
    }
}
''')


cpp_fused_view_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_18 = async_compile.cpp('''
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
    {
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
''')


cpp_fused_sum_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr7[static_cast<long>(x0)];
            auto tmp2 = c10::convert<float>(tmp1);
            auto tmp3 = static_cast<float>(1.1111111111111112);
            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            out_ptr3[static_cast<long>(x0)] = tmp5;
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_23 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        }
    }
}
''')


cpp_fused_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(x1)];
                    auto tmp1 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp2 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp7 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = c10::convert<float>(tmp2);
                    auto tmp4 = static_cast<float>(1.1111111111111112);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = tmp0 ? tmp12 : tmp11;
                    in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp13;
                }
            }
        }
    }
}
''')


cpp_fused_view_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
''')


cpp_fused_sum_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_30 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr7[static_cast<long>(x0)];
            auto tmp2 = c10::convert<float>(tmp1);
            auto tmp3 = static_cast<float>(1.1111111111111112);
            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            out_ptr3[static_cast<long>(x0)] = tmp5;
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        }
    }
}
''')


cpp_fused_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(x1)];
                    auto tmp1 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp2 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp7 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = c10::convert<float>(tmp2);
                    auto tmp4 = static_cast<float>(1.1111111111111112);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = tmp0 ? tmp12 : tmp11;
                    in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp13;
                }
            }
        }
    }
}
''')


cpp_fused_view_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_36 = async_compile.cpp('''
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
    {
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
''')


cpp_fused_sum_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const bool* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr7[static_cast<long>(x0)];
            auto tmp2 = c10::convert<float>(tmp1);
            auto tmp3 = static_cast<float>(1.1111111111111112);
            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            out_ptr3[static_cast<long>(x0)] = tmp5;
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_41 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        }
    }
}
''')


cpp_fused_view_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(x1)];
                    auto tmp1 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp2 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp7 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = c10::convert<float>(tmp2);
                    auto tmp4 = static_cast<float>(1.1111111111111112);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = tmp0 ? tmp12 : tmp11;
                    in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp13;
                }
            }
        }
    }
}
''')


cpp_fused_view_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_45 = async_compile.cpp('''
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
    {
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
''')


cpp_fused_sum_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_48 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr7[static_cast<long>(x0)];
            auto tmp2 = c10::convert<float>(tmp1);
            auto tmp3 = static_cast<float>(1.1111111111111112);
            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            out_ptr3[static_cast<long>(x0)] = tmp5;
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_50 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (3072L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
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
        }
    }
}
''')


cpp_fused_sum_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                        auto tmp6 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(x1)];
                    auto tmp1 = in_out_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp2 = in_ptr1[static_cast<long>(x1 + (128L*x0))];
                    auto tmp7 = in_ptr2[static_cast<long>(x1 + (128L*x0))];
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = c10::convert<float>(tmp2);
                    auto tmp4 = static_cast<float>(1.1111111111111112);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp7)(tmp7 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 - tmp10);
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = tmp0 ? tmp12 : tmp11;
                    in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp13;
                }
            }
        }
    }
}
''')


cpp_fused_view_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_view_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                auto tmp1 = static_cast<float>(8.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_sum_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const bool* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const long* in_ptr8,
                       const long* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr1[static_cast<long>(x0)];
            auto tmp3 = in_ptr2[static_cast<long>(x0)];
            auto tmp5 = in_ptr3[static_cast<long>(x0)];
            auto tmp7 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
            auto tmp8 = c10::convert<float>(tmp7);
            auto tmp9 = static_cast<float>(1.1111111111111112);
            auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
            auto tmp11 = decltype(tmp6)(tmp6 * tmp10);
            in_out_ptr0[static_cast<long>(x0)] = tmp11;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr7[static_cast<long>(x0)];
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp7 = out_ptr1[static_cast<long>(x0)];
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                auto tmp11 = out_ptr2[static_cast<long>(x0)];
                auto tmp17 = in_ptr8[static_cast<long>(x0)];
                auto tmp24 = in_ptr9[static_cast<long>(x0)];
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
                auto tmp25 = static_cast<int>(0);
                auto tmp26 = tmp24 == tmp25;
                auto tmp27 = to_float_mask(tmp26);
                auto tmp28 = decltype(tmp22)::blendv(tmp16, tmp22, tmp27);
                tmp23.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                tmp28.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr7 + static_cast<long>(x0));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
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


cpp_fused_embedding_dense_backward_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(23440896L); x0+=static_cast<long>(8L))
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
    primals_3, primals_13, primals_19, primals_29, primals_35, primals_45, primals_51, primals_61, primals_67, primals_77, primals_83, primals_93, primals_99, primals_104, slice_2, mul, getitem_3, view, view_12, getitem_5, view_17, mul_2, view_19, addmm_4, view_21, getitem_9, mul_7, view_23, getitem_13, view_40, mul_9, view_42, addmm_10, view_44, getitem_17, mul_14, view_46, getitem_21, view_63, mul_16, view_65, addmm_16, view_67, getitem_25, mul_21, view_69, getitem_29, view_86, mul_23, view_88, addmm_22, view_90, getitem_33, mul_28, view_92, getitem_37, view_109, mul_30, view_111, addmm_28, view_113, getitem_41, mul_35, view_115, getitem_45, view_132, mul_37, view_134, addmm_34, view_136, getitem_49, mul_42, getitem_53, view_138, sub_20, ne, sub_22, ne_3, ne_6, where_10, ne_8, where_12, permute_67, div_18, permute_71, permute_75, div_19, permute_79, permute_84, permute_85, alias_10, permute_86, permute_87, permute_90, permute_95, permute_100, div_21, permute_104, permute_108, div_22, permute_112, permute_117, permute_118, alias_11, permute_119, permute_120, permute_123, permute_128, permute_133, div_24, permute_137, permute_141, div_25, permute_145, permute_150, permute_151, alias_12, permute_152, permute_153, permute_156, permute_161, permute_166, div_27, permute_170, permute_174, div_28, permute_178, permute_183, permute_184, alias_13, permute_185, permute_186, permute_189, permute_194, permute_199, div_30, permute_203, permute_207, div_31, permute_211, permute_216, permute_217, alias_14, permute_218, permute_219, permute_222, permute_227, permute_232, div_33, permute_236, permute_240, div_34, permute_244, permute_249, permute_250, alias_15, permute_251, permute_252, permute_255, permute_260, permute_265, div_36, tangents_1, tangents_2, tangents_3 = args
    args.clear()
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_104, (1, 128), (128, 1))
    assert_size_stride(slice_2, (1, 128), (512, 1))
    assert_size_stride(mul, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(getitem_3, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view, (128, 768), (768, 1))
    assert_size_stride(view_12, (1, 1, 1, 128), (128, 128, 128, 1))
    assert_size_stride(getitem_5, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_17, (128, 768), (768, 1))
    assert_size_stride(mul_2, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_19, (128, 768), (768, 1))
    assert_size_stride(addmm_4, (128, 3072), (3072, 1))
    assert_size_stride(view_21, (128, 3072), (3072, 1))
    assert_size_stride(getitem_9, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_7, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_23, (128, 768), (768, 1))
    assert_size_stride(getitem_13, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_40, (128, 768), (768, 1))
    assert_size_stride(mul_9, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_42, (128, 768), (768, 1))
    assert_size_stride(addmm_10, (128, 3072), (3072, 1))
    assert_size_stride(view_44, (128, 3072), (3072, 1))
    assert_size_stride(getitem_17, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_14, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_46, (128, 768), (768, 1))
    assert_size_stride(getitem_21, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_63, (128, 768), (768, 1))
    assert_size_stride(mul_16, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_65, (128, 768), (768, 1))
    assert_size_stride(addmm_16, (128, 3072), (3072, 1))
    assert_size_stride(view_67, (128, 3072), (3072, 1))
    assert_size_stride(getitem_25, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_21, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_69, (128, 768), (768, 1))
    assert_size_stride(getitem_29, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_86, (128, 768), (768, 1))
    assert_size_stride(mul_23, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_88, (128, 768), (768, 1))
    assert_size_stride(addmm_22, (128, 3072), (3072, 1))
    assert_size_stride(view_90, (128, 3072), (3072, 1))
    assert_size_stride(getitem_33, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_28, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_92, (128, 768), (768, 1))
    assert_size_stride(getitem_37, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_109, (128, 768), (768, 1))
    assert_size_stride(mul_30, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_111, (128, 768), (768, 1))
    assert_size_stride(addmm_28, (128, 3072), (3072, 1))
    assert_size_stride(view_113, (128, 3072), (3072, 1))
    assert_size_stride(getitem_41, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_35, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_115, (128, 768), (768, 1))
    assert_size_stride(getitem_45, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(view_132, (128, 768), (768, 1))
    assert_size_stride(mul_37, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_134, (128, 768), (768, 1))
    assert_size_stride(addmm_34, (128, 3072), (3072, 1))
    assert_size_stride(view_136, (128, 3072), (3072, 1))
    assert_size_stride(getitem_49, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(mul_42, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(getitem_53, (1, 128, 768), (98304, 768, 1))
    assert_size_stride(view_138, (128, 768), (768, 1))
    assert_size_stride(sub_20, (1, 128), (128, 1))
    assert_size_stride(ne, (1, ), (1, ))
    assert_size_stride(sub_22, (1, 128), (128, 1))
    assert_size_stride(ne_3, (1, ), (1, ))
    assert_size_stride(ne_6, (1, 1), (1, 1))
    assert_size_stride(where_10, (1, 1), (1, 1))
    assert_size_stride(ne_8, (1, 1), (1, 1))
    assert_size_stride(where_12, (1, 1), (1, 1))
    assert_size_stride(permute_67, (2, 768), (768, 1))
    assert_size_stride(div_18, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_71, (768, 3072), (3072, 1))
    assert_size_stride(permute_75, (3072, 768), (768, 1))
    assert_size_stride(div_19, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_79, (768, 768), (768, 1))
    assert_size_stride(permute_84, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_85, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_10, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_86, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_87, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_90, (768, 768), (768, 1))
    assert_size_stride(permute_95, (768, 768), (768, 1))
    assert_size_stride(permute_100, (768, 768), (768, 1))
    assert_size_stride(div_21, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_104, (768, 3072), (3072, 1))
    assert_size_stride(permute_108, (3072, 768), (768, 1))
    assert_size_stride(div_22, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_112, (768, 768), (768, 1))
    assert_size_stride(permute_117, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_118, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_11, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_119, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_120, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_123, (768, 768), (768, 1))
    assert_size_stride(permute_128, (768, 768), (768, 1))
    assert_size_stride(permute_133, (768, 768), (768, 1))
    assert_size_stride(div_24, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_137, (768, 3072), (3072, 1))
    assert_size_stride(permute_141, (3072, 768), (768, 1))
    assert_size_stride(div_25, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_145, (768, 768), (768, 1))
    assert_size_stride(permute_150, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_151, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_12, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_152, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_153, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_156, (768, 768), (768, 1))
    assert_size_stride(permute_161, (768, 768), (768, 1))
    assert_size_stride(permute_166, (768, 768), (768, 1))
    assert_size_stride(div_27, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_170, (768, 3072), (3072, 1))
    assert_size_stride(permute_174, (3072, 768), (768, 1))
    assert_size_stride(div_28, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_178, (768, 768), (768, 1))
    assert_size_stride(permute_183, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_184, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_13, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_185, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_186, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_189, (768, 768), (768, 1))
    assert_size_stride(permute_194, (768, 768), (768, 1))
    assert_size_stride(permute_199, (768, 768), (768, 1))
    assert_size_stride(div_30, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_203, (768, 3072), (3072, 1))
    assert_size_stride(permute_207, (3072, 768), (768, 1))
    assert_size_stride(div_31, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_211, (768, 768), (768, 1))
    assert_size_stride(permute_216, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_217, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_14, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_218, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_219, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_222, (768, 768), (768, 1))
    assert_size_stride(permute_227, (768, 768), (768, 1))
    assert_size_stride(permute_232, (768, 768), (768, 1))
    assert_size_stride(div_33, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_236, (768, 3072), (3072, 1))
    assert_size_stride(permute_240, (3072, 768), (768, 1))
    assert_size_stride(div_34, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_244, (768, 768), (768, 1))
    assert_size_stride(permute_249, (12, 128, 128), (16384, 1, 128))
    assert_size_stride(permute_250, (12, 64, 128), (64, 1, 768))
    assert_size_stride(alias_15, (1, 12, 128, 128), (196608, 16384, 128, 1))
    assert_size_stride(permute_251, (12, 64, 128), (64, 1, 768))
    assert_size_stride(permute_252, (12, 128, 64), (64, 768, 1))
    assert_size_stride(permute_255, (768, 768), (768, 1))
    assert_size_stride(permute_260, (768, 768), (768, 1))
    assert_size_stride(permute_265, (768, 768), (768, 1))
    assert_size_stride(div_36, (1, 128, 1), (128, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (1, 128), (128, 1))
    assert_size_stride(tangents_3, (1, 128), (128, 1))
    buf0 = empty((1, 128), device='cpu', dtype=torch.float32)
    cpp_fused_nll_loss_backward_0(c_void_p(buf0.data_ptr()))
    aten.scatter_(buf0,1,where_10,-1.0)
    del where_10
    buf4 = empty((1, 128), device='cpu', dtype=torch.float32)
    cpp_fused_nll_loss_backward_1(c_void_p(buf4.data_ptr()))
    aten.scatter_(buf4,1,where_12,-1.0)
    del where_12
    buf3 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf7 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf8 = empty((1, 128, 2), device='cpu', dtype=torch.float32)
    cpp_fused__log_softmax_backward_data_cat_div_nll_loss_backward_nll_loss_forward_2(c_void_p(buf0.data_ptr()), c_void_p(ne_6.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(ne_3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(ne_8.data_ptr()), c_void_p(ne.data_ptr()), c_void_p(tangents_2.data_ptr()), c_void_p(sub_20.data_ptr()), c_void_p(tangents_3.data_ptr()), c_void_p(sub_22.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()))
    del buf3
    del buf7
    del ne
    del ne_3
    del ne_6
    del ne_8
    del sub_20
    del sub_22
    del tangents_1
    del tangents_2
    del tangents_3
    buf9 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf8, (128, 2), (2, 1), 0), permute_67, out=buf9)
    del permute_67
    buf10 = empty((2, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf8, (2, 128), (1, 2), 0), view_138, out=buf10)
    del view_138
    buf11 = empty((1, 2), device='cpu', dtype=torch.float32)
    buf12 = reinterpret_tensor(buf4, (1, 128, 1), (128, 1, 128), 0); del buf4  # reuse
    buf13 = reinterpret_tensor(buf0, (1, 128, 1), (128, 1, 128), 0); del buf0  # reuse
    buf14 = empty((1, 128, 768), device='cpu', dtype=torch.float32)
    buf15 = empty((768, ), device='cpu', dtype=torch.float32)
    buf16 = empty((768, ), device='cpu', dtype=torch.float32)
    buf17 = empty((1, 128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_native_dropout_backward_native_layer_norm_backward_sum_3(c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(getitem_53.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(mul_42.data_ptr()), c_void_p(div_18.data_ptr()), c_void_p(getitem_49.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()))
    del buf8
    del div_18
    del getitem_49
    del getitem_53
    del mul_42
    del primals_99
    buf18 = empty((128, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf17, (128, 768), (768, 1), 0), permute_71, out=buf18)
    del permute_71
    buf19 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf17, (768, 128), (1, 768), 0), view_136, out=buf19)
    del view_136
    buf20 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf21 = reinterpret_tensor(buf18, (1, 128, 3072), (393216, 3072, 1), 0); del buf18  # reuse
    cpp_fused_gelu_gelu_backward_sum_4(c_void_p(buf21.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(buf20.data_ptr()))
    del addmm_34
    buf22 = reinterpret_tensor(buf17, (128, 768), (768, 1), 0); del buf17  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf21, (128, 3072), (3072, 1), 0), permute_75, out=buf22)
    del permute_75
    buf23 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf21, (3072, 128), (1, 3072), 0), view_134, out=buf23)
    del view_134
    buf24 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf25 = buf13; del buf13  # reuse
    buf26 = buf12; del buf12  # reuse
    buf27 = reinterpret_tensor(buf9, (1, 128, 768), (98304, 768, 1), 0); del buf9  # reuse
    buf28 = empty((768, ), device='cpu', dtype=torch.float32)
    buf29 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_5(c_void_p(buf21.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(mul_37.data_ptr()), c_void_p(div_19.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()))
    del div_19
    del mul_37
    del primals_93
    buf30 = buf22; del buf22  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf27, (128, 768), (768, 1), 0), permute_79, out=buf30)
    del permute_79
    buf31 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf27, (768, 128), (1, 768), 0), view_132, out=buf31)
    del view_132
    buf33 = reinterpret_tensor(buf14, (12, 128, 64), (8192, 64, 1), 0); del buf14  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_84, reinterpret_tensor(buf30, (12, 128, 64), (64, 768, 1), 0), out=buf33)
    del permute_84
    buf39 = empty((128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_6(c_void_p(buf33.data_ptr()), c_void_p(buf39.data_ptr()))
    buf40 = reinterpret_tensor(buf33, (128, 768), (768, 1), 0); del buf33  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf39, permute_90, out=buf40)
    del permute_90
    buf34 = empty((12, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf30, (12, 128, 64), (64, 768, 1), 0), permute_85, out=buf34)
    del permute_85
    buf35 = empty_strided((1, 12, 128, 1), (1536, 128, 1, 1536), device='cpu', dtype=torch.float32)
    buf36 = reinterpret_tensor(buf34, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf34  # reuse
    cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_7(c_void_p(buf36.data_ptr()), c_void_p(getitem_45.data_ptr()), c_void_p(alias_10.data_ptr()), c_void_p(view_12.data_ptr()), c_void_p(buf35.data_ptr()))
    del alias_10
    del getitem_45
    buf37 = reinterpret_tensor(buf30, (12, 64, 128), (8192, 128, 1), 0); del buf30  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_86, reinterpret_tensor(buf36, (12, 128, 128), (16384, 128, 1), 0), out=buf37)
    del permute_86
    buf43 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf37, (128, 768), (1, 128), 0), permute_95, out=buf43)
    del permute_95
    buf38 = empty((12, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf36, (12, 128, 128), (16384, 128, 1), 0), permute_87, out=buf38)
    del permute_87
    buf46 = empty((128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_8(c_void_p(buf38.data_ptr()), c_void_p(buf46.data_ptr()))
    buf47 = reinterpret_tensor(buf38, (128, 768), (768, 1), 0); del buf38  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf46, permute_100, out=buf47)
    del permute_100
    buf32 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf54 = empty((768, ), device='cpu', dtype=torch.float32)
    buf55 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_9(c_void_p(buf27.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(mul_35.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()))
    buf41 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf39, (768, 128), (1, 768), 0), view_115, out=buf41)
    buf42 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_10(c_void_p(buf39.data_ptr()), c_void_p(buf42.data_ptr()))
    buf44 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf37, (768, 128), (128, 1), 0), view_115, out=buf44)
    buf45 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_11(c_void_p(buf37.data_ptr()), c_void_p(buf45.data_ptr()))
    buf48 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf46, (768, 128), (1, 768), 0), view_115, out=buf48)
    del view_115
    buf49 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf50 = buf27; del buf27  # reuse
    buf51 = buf26; del buf26  # reuse
    buf52 = buf25; del buf25  # reuse
    buf53 = buf50; del buf50  # reuse
    buf56 = reinterpret_tensor(buf37, (1, 128, 768), (98304, 768, 1), 0); del buf37  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_12(c_void_p(buf53.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(mul_35.data_ptr()), c_void_p(div_21.data_ptr()), c_void_p(getitem_41.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf56.data_ptr()))
    del div_21
    del getitem_41
    del mul_35
    del primals_83
    buf57 = reinterpret_tensor(buf21, (128, 3072), (3072, 1), 0); del buf21  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf56, (128, 768), (768, 1), 0), permute_104, out=buf57)
    del permute_104
    buf58 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf56, (768, 128), (1, 768), 0), view_113, out=buf58)
    del view_113
    buf59 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf60 = reinterpret_tensor(buf57, (1, 128, 3072), (393216, 3072, 1), 0); del buf57  # reuse
    cpp_fused_gelu_gelu_backward_sum_13(c_void_p(buf60.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(addmm_28.data_ptr()), c_void_p(buf59.data_ptr()))
    del addmm_28
    buf61 = reinterpret_tensor(buf56, (128, 768), (768, 1), 0); del buf56  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf60, (128, 3072), (3072, 1), 0), permute_108, out=buf61)
    del permute_108
    buf62 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf60, (3072, 128), (1, 3072), 0), view_111, out=buf62)
    del view_111
    buf63 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf64 = buf52; del buf52  # reuse
    buf65 = buf51; del buf51  # reuse
    buf66 = reinterpret_tensor(buf47, (1, 128, 768), (98304, 768, 1), 0); del buf47  # reuse
    buf67 = empty((768, ), device='cpu', dtype=torch.float32)
    buf68 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_14(c_void_p(buf60.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(mul_30.data_ptr()), c_void_p(div_22.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()))
    del div_22
    del mul_30
    del primals_77
    buf69 = buf61; del buf61  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf66, (128, 768), (768, 1), 0), permute_112, out=buf69)
    del permute_112
    buf70 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf66, (768, 128), (1, 768), 0), view_109, out=buf70)
    del view_109
    buf72 = reinterpret_tensor(buf53, (12, 128, 64), (8192, 64, 1), 0); del buf53  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_117, reinterpret_tensor(buf69, (12, 128, 64), (64, 768, 1), 0), out=buf72)
    del permute_117
    buf78 = buf46; del buf46  # reuse
    cpp_fused_view_15(c_void_p(buf72.data_ptr()), c_void_p(buf78.data_ptr()))
    buf79 = reinterpret_tensor(buf72, (128, 768), (768, 1), 0); del buf72  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf78, permute_123, out=buf79)
    del permute_123
    buf73 = reinterpret_tensor(buf36, (12, 128, 128), (16384, 128, 1), 0); del buf36  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf69, (12, 128, 64), (64, 768, 1), 0), permute_118, out=buf73)
    del permute_118
    buf74 = buf35; del buf35  # reuse
    buf75 = reinterpret_tensor(buf73, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf73  # reuse
    cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_16(c_void_p(buf75.data_ptr()), c_void_p(getitem_37.data_ptr()), c_void_p(alias_11.data_ptr()), c_void_p(view_12.data_ptr()), c_void_p(buf74.data_ptr()))
    del alias_11
    del getitem_37
    buf76 = reinterpret_tensor(buf69, (12, 64, 128), (8192, 128, 1), 0); del buf69  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_119, reinterpret_tensor(buf75, (12, 128, 128), (16384, 128, 1), 0), out=buf76)
    del permute_119
    buf82 = buf43; del buf43  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf76, (128, 768), (1, 128), 0), permute_128, out=buf82)
    del permute_128
    buf77 = reinterpret_tensor(buf40, (12, 128, 64), (8192, 64, 1), 0); del buf40  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf75, (12, 128, 128), (16384, 128, 1), 0), permute_120, out=buf77)
    del permute_120
    buf85 = buf39; del buf39  # reuse
    cpp_fused_view_17(c_void_p(buf77.data_ptr()), c_void_p(buf85.data_ptr()))
    buf86 = reinterpret_tensor(buf77, (128, 768), (768, 1), 0); del buf77  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf85, permute_133, out=buf86)
    del permute_133
    buf71 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf93 = empty((768, ), device='cpu', dtype=torch.float32)
    buf94 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_18(c_void_p(buf66.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(mul_28.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()))
    buf80 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf78, (768, 128), (1, 768), 0), view_92, out=buf80)
    buf81 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_19(c_void_p(buf78.data_ptr()), c_void_p(buf81.data_ptr()))
    buf83 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf76, (768, 128), (128, 1), 0), view_92, out=buf83)
    buf84 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_20(c_void_p(buf76.data_ptr()), c_void_p(buf84.data_ptr()))
    buf87 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf85, (768, 128), (1, 768), 0), view_92, out=buf87)
    del view_92
    buf88 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf89 = buf66; del buf66  # reuse
    buf90 = buf65; del buf65  # reuse
    buf91 = buf64; del buf64  # reuse
    buf92 = buf89; del buf89  # reuse
    buf95 = reinterpret_tensor(buf76, (1, 128, 768), (98304, 768, 1), 0); del buf76  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_21(c_void_p(buf92.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(mul_28.data_ptr()), c_void_p(div_24.data_ptr()), c_void_p(getitem_33.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf95.data_ptr()))
    del div_24
    del getitem_33
    del mul_28
    del primals_67
    buf96 = reinterpret_tensor(buf60, (128, 3072), (3072, 1), 0); del buf60  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf95, (128, 768), (768, 1), 0), permute_137, out=buf96)
    del permute_137
    buf97 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf95, (768, 128), (1, 768), 0), view_90, out=buf97)
    del view_90
    buf98 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf99 = reinterpret_tensor(buf96, (1, 128, 3072), (393216, 3072, 1), 0); del buf96  # reuse
    cpp_fused_gelu_gelu_backward_sum_22(c_void_p(buf99.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(buf98.data_ptr()))
    del addmm_22
    buf100 = reinterpret_tensor(buf95, (128, 768), (768, 1), 0); del buf95  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf99, (128, 3072), (3072, 1), 0), permute_141, out=buf100)
    del permute_141
    buf101 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf99, (3072, 128), (1, 3072), 0), view_88, out=buf101)
    del view_88
    buf102 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf103 = buf91; del buf91  # reuse
    buf104 = buf90; del buf90  # reuse
    buf105 = reinterpret_tensor(buf86, (1, 128, 768), (98304, 768, 1), 0); del buf86  # reuse
    buf106 = empty((768, ), device='cpu', dtype=torch.float32)
    buf107 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_23(c_void_p(buf99.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(mul_23.data_ptr()), c_void_p(div_25.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()))
    del div_25
    del mul_23
    del primals_61
    buf108 = reinterpret_tensor(buf92, (128, 768), (768, 1), 0); del buf92  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf105, (128, 768), (768, 1), 0), permute_145, out=buf108)
    del permute_145
    buf109 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf105, (768, 128), (1, 768), 0), view_86, out=buf109)
    del view_86
    buf111 = reinterpret_tensor(buf100, (12, 128, 64), (8192, 64, 1), 0); del buf100  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_150, reinterpret_tensor(buf108, (12, 128, 64), (64, 768, 1), 0), out=buf111)
    del permute_150
    buf117 = buf85; del buf85  # reuse
    cpp_fused_view_24(c_void_p(buf111.data_ptr()), c_void_p(buf117.data_ptr()))
    buf118 = reinterpret_tensor(buf111, (128, 768), (768, 1), 0); del buf111  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf117, permute_156, out=buf118)
    del permute_156
    buf112 = reinterpret_tensor(buf75, (12, 128, 128), (16384, 128, 1), 0); del buf75  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf108, (12, 128, 64), (64, 768, 1), 0), permute_151, out=buf112)
    del permute_151
    buf113 = buf74; del buf74  # reuse
    buf114 = reinterpret_tensor(buf112, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf112  # reuse
    cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_25(c_void_p(buf114.data_ptr()), c_void_p(getitem_29.data_ptr()), c_void_p(alias_12.data_ptr()), c_void_p(view_12.data_ptr()), c_void_p(buf113.data_ptr()))
    del alias_12
    del getitem_29
    buf115 = reinterpret_tensor(buf108, (12, 64, 128), (8192, 128, 1), 0); del buf108  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_152, reinterpret_tensor(buf114, (12, 128, 128), (16384, 128, 1), 0), out=buf115)
    del permute_152
    buf121 = buf82; del buf82  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf115, (128, 768), (1, 128), 0), permute_161, out=buf121)
    del permute_161
    buf116 = reinterpret_tensor(buf79, (12, 128, 64), (8192, 64, 1), 0); del buf79  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf114, (12, 128, 128), (16384, 128, 1), 0), permute_153, out=buf116)
    del permute_153
    buf124 = buf78; del buf78  # reuse
    cpp_fused_view_26(c_void_p(buf116.data_ptr()), c_void_p(buf124.data_ptr()))
    buf125 = reinterpret_tensor(buf116, (128, 768), (768, 1), 0); del buf116  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf124, permute_166, out=buf125)
    del permute_166
    buf110 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf132 = empty((768, ), device='cpu', dtype=torch.float32)
    buf133 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_27(c_void_p(buf105.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(mul_21.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()))
    buf119 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf117, (768, 128), (1, 768), 0), view_69, out=buf119)
    buf120 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_28(c_void_p(buf117.data_ptr()), c_void_p(buf120.data_ptr()))
    buf122 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf115, (768, 128), (128, 1), 0), view_69, out=buf122)
    buf123 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_29(c_void_p(buf115.data_ptr()), c_void_p(buf123.data_ptr()))
    buf126 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf124, (768, 128), (1, 768), 0), view_69, out=buf126)
    del view_69
    buf127 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf128 = buf105; del buf105  # reuse
    buf129 = buf104; del buf104  # reuse
    buf130 = buf103; del buf103  # reuse
    buf131 = buf128; del buf128  # reuse
    buf134 = reinterpret_tensor(buf115, (1, 128, 768), (98304, 768, 1), 0); del buf115  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_30(c_void_p(buf131.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(mul_21.data_ptr()), c_void_p(div_27.data_ptr()), c_void_p(getitem_25.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf134.data_ptr()))
    del div_27
    del getitem_25
    del mul_21
    del primals_51
    buf135 = reinterpret_tensor(buf99, (128, 3072), (3072, 1), 0); del buf99  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf134, (128, 768), (768, 1), 0), permute_170, out=buf135)
    del permute_170
    buf136 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf134, (768, 128), (1, 768), 0), view_67, out=buf136)
    del view_67
    buf137 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf138 = reinterpret_tensor(buf135, (1, 128, 3072), (393216, 3072, 1), 0); del buf135  # reuse
    cpp_fused_gelu_gelu_backward_sum_31(c_void_p(buf138.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(addmm_16.data_ptr()), c_void_p(buf137.data_ptr()))
    del addmm_16
    buf139 = reinterpret_tensor(buf134, (128, 768), (768, 1), 0); del buf134  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf138, (128, 3072), (3072, 1), 0), permute_174, out=buf139)
    del permute_174
    buf140 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf138, (3072, 128), (1, 3072), 0), view_65, out=buf140)
    del view_65
    buf141 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf142 = buf130; del buf130  # reuse
    buf143 = buf129; del buf129  # reuse
    buf144 = reinterpret_tensor(buf125, (1, 128, 768), (98304, 768, 1), 0); del buf125  # reuse
    buf145 = empty((768, ), device='cpu', dtype=torch.float32)
    buf146 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_32(c_void_p(buf138.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(mul_16.data_ptr()), c_void_p(div_28.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()))
    del div_28
    del mul_16
    del primals_45
    buf147 = buf139; del buf139  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf144, (128, 768), (768, 1), 0), permute_178, out=buf147)
    del permute_178
    buf148 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf144, (768, 128), (1, 768), 0), view_63, out=buf148)
    del view_63
    buf150 = reinterpret_tensor(buf131, (12, 128, 64), (8192, 64, 1), 0); del buf131  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_183, reinterpret_tensor(buf147, (12, 128, 64), (64, 768, 1), 0), out=buf150)
    del permute_183
    buf156 = buf124; del buf124  # reuse
    cpp_fused_view_33(c_void_p(buf150.data_ptr()), c_void_p(buf156.data_ptr()))
    buf157 = reinterpret_tensor(buf150, (128, 768), (768, 1), 0); del buf150  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf156, permute_189, out=buf157)
    del permute_189
    buf151 = reinterpret_tensor(buf114, (12, 128, 128), (16384, 128, 1), 0); del buf114  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf147, (12, 128, 64), (64, 768, 1), 0), permute_184, out=buf151)
    del permute_184
    buf152 = buf113; del buf113  # reuse
    buf153 = reinterpret_tensor(buf151, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf151  # reuse
    cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_34(c_void_p(buf153.data_ptr()), c_void_p(getitem_21.data_ptr()), c_void_p(alias_13.data_ptr()), c_void_p(view_12.data_ptr()), c_void_p(buf152.data_ptr()))
    del alias_13
    del getitem_21
    buf154 = reinterpret_tensor(buf147, (12, 64, 128), (8192, 128, 1), 0); del buf147  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_185, reinterpret_tensor(buf153, (12, 128, 128), (16384, 128, 1), 0), out=buf154)
    del permute_185
    buf160 = buf121; del buf121  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf154, (128, 768), (1, 128), 0), permute_194, out=buf160)
    del permute_194
    buf155 = reinterpret_tensor(buf118, (12, 128, 64), (8192, 64, 1), 0); del buf118  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf153, (12, 128, 128), (16384, 128, 1), 0), permute_186, out=buf155)
    del permute_186
    buf163 = buf117; del buf117  # reuse
    cpp_fused_view_35(c_void_p(buf155.data_ptr()), c_void_p(buf163.data_ptr()))
    buf164 = reinterpret_tensor(buf155, (128, 768), (768, 1), 0); del buf155  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf163, permute_199, out=buf164)
    del permute_199
    buf149 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf171 = empty((768, ), device='cpu', dtype=torch.float32)
    buf172 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_36(c_void_p(buf144.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(mul_14.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()))
    buf158 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf156, (768, 128), (1, 768), 0), view_46, out=buf158)
    buf159 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_37(c_void_p(buf156.data_ptr()), c_void_p(buf159.data_ptr()))
    buf161 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf154, (768, 128), (128, 1), 0), view_46, out=buf161)
    buf162 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_38(c_void_p(buf154.data_ptr()), c_void_p(buf162.data_ptr()))
    buf165 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf163, (768, 128), (1, 768), 0), view_46, out=buf165)
    del view_46
    buf166 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf167 = buf144; del buf144  # reuse
    buf168 = buf143; del buf143  # reuse
    buf169 = buf142; del buf142  # reuse
    buf170 = buf167; del buf167  # reuse
    buf173 = reinterpret_tensor(buf154, (1, 128, 768), (98304, 768, 1), 0); del buf154  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_39(c_void_p(buf170.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(mul_14.data_ptr()), c_void_p(div_30.data_ptr()), c_void_p(getitem_17.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf173.data_ptr()))
    del div_30
    del getitem_17
    del mul_14
    del primals_35
    buf174 = reinterpret_tensor(buf138, (128, 3072), (3072, 1), 0); del buf138  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf173, (128, 768), (768, 1), 0), permute_203, out=buf174)
    del permute_203
    buf175 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf173, (768, 128), (1, 768), 0), view_44, out=buf175)
    del view_44
    buf176 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf177 = reinterpret_tensor(buf174, (1, 128, 3072), (393216, 3072, 1), 0); del buf174  # reuse
    cpp_fused_gelu_gelu_backward_sum_40(c_void_p(buf177.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(buf176.data_ptr()))
    del addmm_10
    buf178 = reinterpret_tensor(buf173, (128, 768), (768, 1), 0); del buf173  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf177, (128, 3072), (3072, 1), 0), permute_207, out=buf178)
    del permute_207
    buf179 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf177, (3072, 128), (1, 3072), 0), view_42, out=buf179)
    del view_42
    buf180 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf181 = buf169; del buf169  # reuse
    buf182 = buf168; del buf168  # reuse
    buf183 = reinterpret_tensor(buf164, (1, 128, 768), (98304, 768, 1), 0); del buf164  # reuse
    buf184 = empty((768, ), device='cpu', dtype=torch.float32)
    buf185 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_41(c_void_p(buf177.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(mul_9.data_ptr()), c_void_p(div_31.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()))
    del div_31
    del mul_9
    del primals_29
    buf186 = buf178; del buf178  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf183, (128, 768), (768, 1), 0), permute_211, out=buf186)
    del permute_211
    buf187 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf183, (768, 128), (1, 768), 0), view_40, out=buf187)
    del view_40
    buf189 = reinterpret_tensor(buf170, (12, 128, 64), (8192, 64, 1), 0); del buf170  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_216, reinterpret_tensor(buf186, (12, 128, 64), (64, 768, 1), 0), out=buf189)
    del permute_216
    buf195 = buf163; del buf163  # reuse
    cpp_fused_view_42(c_void_p(buf189.data_ptr()), c_void_p(buf195.data_ptr()))
    buf196 = reinterpret_tensor(buf189, (128, 768), (768, 1), 0); del buf189  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf195, permute_222, out=buf196)
    del permute_222
    buf190 = reinterpret_tensor(buf153, (12, 128, 128), (16384, 128, 1), 0); del buf153  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf186, (12, 128, 64), (64, 768, 1), 0), permute_217, out=buf190)
    del permute_217
    buf191 = buf152; del buf152  # reuse
    buf192 = reinterpret_tensor(buf190, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf190  # reuse
    cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_43(c_void_p(buf192.data_ptr()), c_void_p(getitem_13.data_ptr()), c_void_p(alias_14.data_ptr()), c_void_p(view_12.data_ptr()), c_void_p(buf191.data_ptr()))
    del alias_14
    del getitem_13
    buf193 = reinterpret_tensor(buf186, (12, 64, 128), (8192, 128, 1), 0); del buf186  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_218, reinterpret_tensor(buf192, (12, 128, 128), (16384, 128, 1), 0), out=buf193)
    del permute_218
    buf199 = buf160; del buf160  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf193, (128, 768), (1, 128), 0), permute_227, out=buf199)
    del permute_227
    buf194 = reinterpret_tensor(buf157, (12, 128, 64), (8192, 64, 1), 0); del buf157  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf192, (12, 128, 128), (16384, 128, 1), 0), permute_219, out=buf194)
    del permute_219
    buf202 = buf156; del buf156  # reuse
    cpp_fused_view_44(c_void_p(buf194.data_ptr()), c_void_p(buf202.data_ptr()))
    buf203 = reinterpret_tensor(buf194, (128, 768), (768, 1), 0); del buf194  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf202, permute_232, out=buf203)
    del permute_232
    buf188 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf210 = empty((768, ), device='cpu', dtype=torch.float32)
    buf211 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_45(c_void_p(buf183.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(mul_7.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()))
    buf197 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf195, (768, 128), (1, 768), 0), view_23, out=buf197)
    buf198 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_46(c_void_p(buf195.data_ptr()), c_void_p(buf198.data_ptr()))
    buf200 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf193, (768, 128), (128, 1), 0), view_23, out=buf200)
    buf201 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_47(c_void_p(buf193.data_ptr()), c_void_p(buf201.data_ptr()))
    buf204 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf202, (768, 128), (1, 768), 0), view_23, out=buf204)
    del view_23
    buf205 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf206 = buf183; del buf183  # reuse
    buf207 = buf182; del buf182  # reuse
    buf208 = buf181; del buf181  # reuse
    buf209 = buf206; del buf206  # reuse
    buf212 = reinterpret_tensor(buf193, (1, 128, 768), (98304, 768, 1), 0); del buf193  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_48(c_void_p(buf209.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(mul_7.data_ptr()), c_void_p(div_33.data_ptr()), c_void_p(getitem_9.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf212.data_ptr()))
    del div_33
    del getitem_9
    del mul_7
    del primals_19
    buf213 = reinterpret_tensor(buf177, (128, 3072), (3072, 1), 0); del buf177  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf212, (128, 768), (768, 1), 0), permute_236, out=buf213)
    del permute_236
    buf214 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf212, (768, 128), (1, 768), 0), view_21, out=buf214)
    del view_21
    buf215 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf216 = reinterpret_tensor(buf213, (1, 128, 3072), (393216, 3072, 1), 0); del buf213  # reuse
    cpp_fused_gelu_gelu_backward_sum_49(c_void_p(buf216.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(addmm_4.data_ptr()), c_void_p(buf215.data_ptr()))
    del addmm_4
    buf217 = reinterpret_tensor(buf212, (128, 768), (768, 1), 0); del buf212  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf216, (128, 3072), (3072, 1), 0), permute_240, out=buf217)
    del permute_240
    buf218 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf216, (3072, 128), (1, 3072), 0), view_19, out=buf218)
    del view_19
    buf219 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf220 = buf208; del buf208  # reuse
    buf221 = buf207; del buf207  # reuse
    buf222 = reinterpret_tensor(buf203, (1, 128, 768), (98304, 768, 1), 0); del buf203  # reuse
    buf223 = empty((768, ), device='cpu', dtype=torch.float32)
    buf224 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_50(c_void_p(buf216.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(mul_2.data_ptr()), c_void_p(div_34.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()))
    del div_34
    del mul_2
    del primals_13
    buf225 = buf217; del buf217  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf222, (128, 768), (768, 1), 0), permute_244, out=buf225)
    del permute_244
    buf226 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf222, (768, 128), (1, 768), 0), view_17, out=buf226)
    del view_17
    buf227 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_51(c_void_p(buf222.data_ptr()), c_void_p(buf227.data_ptr()))
    buf228 = reinterpret_tensor(buf209, (12, 128, 64), (8192, 64, 1), 0); del buf209  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_249, reinterpret_tensor(buf225, (12, 128, 64), (64, 768, 1), 0), out=buf228)
    del permute_249
    buf229 = reinterpret_tensor(buf192, (12, 128, 128), (16384, 128, 1), 0); del buf192  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf225, (12, 128, 64), (64, 768, 1), 0), permute_250, out=buf229)
    del permute_250
    buf230 = buf191; del buf191  # reuse
    buf231 = reinterpret_tensor(buf229, (1, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf229  # reuse
    cpp_fused__softmax_backward_data_masked_fill_native_dropout_backward_nll_loss_forward_52(c_void_p(buf231.data_ptr()), c_void_p(getitem_5.data_ptr()), c_void_p(alias_15.data_ptr()), c_void_p(view_12.data_ptr()), c_void_p(buf230.data_ptr()))
    del alias_15
    del buf230
    del getitem_5
    del view_12
    buf232 = reinterpret_tensor(buf225, (12, 64, 128), (8192, 128, 1), 0); del buf225  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_251, reinterpret_tensor(buf231, (12, 128, 128), (16384, 128, 1), 0), out=buf232)
    del permute_251
    buf233 = reinterpret_tensor(buf202, (12, 128, 64), (8192, 64, 1), 0); del buf202  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf231, (12, 128, 128), (16384, 128, 1), 0), permute_252, out=buf233)
    del buf231
    del permute_252
    buf234 = buf199; del buf199  # reuse
    cpp_fused_view_53(c_void_p(buf228.data_ptr()), c_void_p(buf234.data_ptr()))
    buf235 = reinterpret_tensor(buf228, (128, 768), (768, 1), 0); del buf228  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf234, permute_255, out=buf235)
    del permute_255
    buf236 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf234, (768, 128), (1, 768), 0), view, out=buf236)
    buf237 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_54(c_void_p(buf234.data_ptr()), c_void_p(buf237.data_ptr()))
    buf238 = buf234; del buf234  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf232, (128, 768), (1, 128), 0), permute_260, out=buf238)
    del permute_260
    buf239 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf232, (768, 128), (128, 1), 0), view, out=buf239)
    buf240 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf241 = buf196; del buf196  # reuse
    cpp_fused_sum_view_55(c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()))
    buf242 = reinterpret_tensor(buf233, (128, 768), (768, 1), 0); del buf233  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf241, permute_265, out=buf242)
    del permute_265
    buf243 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf241, (768, 128), (1, 768), 0), view, out=buf243)
    del view
    buf244 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf245 = buf222; del buf222  # reuse
    buf246 = buf221; del buf221  # reuse
    buf247 = buf220; del buf220  # reuse
    buf252 = reinterpret_tensor(buf232, (1, 128, 768), (98304, 768, 1), 0); del buf232  # reuse
    buf256 = reinterpret_tensor(buf195, (1, 128, 768), (98304, 768, 1), 0); del buf195  # reuse
    buf249 = empty((768, ), device='cpu', dtype=torch.float32)
    buf250 = empty((768, ), device='cpu', dtype=torch.float32)
    buf251 = reinterpret_tensor(buf216, (512, 768), (768, 1), 0); del buf216  # reuse
    cpp_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_sum_56(c_void_p(buf245.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(getitem_3.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(mul.data_ptr()), c_void_p(div_36.data_ptr()), c_void_p(slice_2.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()))
    del buf235
    del buf238
    del buf241
    del buf242
    del buf245
    del buf246
    del buf247
    del div_36
    del getitem_3
    del mul
    del primals_3
    aten.index_put_(buf251, [slice_2], buf252, True)
    del buf252
    del slice_2
    buf255 = empty((30522, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_57(c_void_p(buf255.data_ptr()))
    aten.index_put_(buf255, [primals_104], buf256, True)
    del buf256
    del primals_104
    return (buf255, buf251, buf249, buf250, reinterpret_tensor(buf243, (768, 768), (768, 1), 0), reinterpret_tensor(buf244, (768, ), (1, ), 0), reinterpret_tensor(buf239, (768, 768), (768, 1), 0), reinterpret_tensor(buf240, (768, ), (1, ), 0), reinterpret_tensor(buf236, (768, 768), (768, 1), 0), reinterpret_tensor(buf237, (768, ), (1, ), 0), reinterpret_tensor(buf226, (768, 768), (768, 1), 0), reinterpret_tensor(buf227, (768, ), (1, ), 0), buf223, buf224, reinterpret_tensor(buf218, (3072, 768), (768, 1), 0), reinterpret_tensor(buf219, (3072, ), (1, ), 0), reinterpret_tensor(buf214, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf215, (768, ), (1, ), 0), buf210, buf211, reinterpret_tensor(buf204, (768, 768), (768, 1), 0), reinterpret_tensor(buf205, (768, ), (1, ), 0), reinterpret_tensor(buf200, (768, 768), (768, 1), 0), reinterpret_tensor(buf201, (768, ), (1, ), 0), reinterpret_tensor(buf197, (768, 768), (768, 1), 0), reinterpret_tensor(buf198, (768, ), (1, ), 0), reinterpret_tensor(buf187, (768, 768), (768, 1), 0), reinterpret_tensor(buf188, (768, ), (1, ), 0), buf184, buf185, reinterpret_tensor(buf179, (3072, 768), (768, 1), 0), reinterpret_tensor(buf180, (3072, ), (1, ), 0), reinterpret_tensor(buf175, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf176, (768, ), (1, ), 0), buf171, buf172, reinterpret_tensor(buf165, (768, 768), (768, 1), 0), reinterpret_tensor(buf166, (768, ), (1, ), 0), reinterpret_tensor(buf161, (768, 768), (768, 1), 0), reinterpret_tensor(buf162, (768, ), (1, ), 0), reinterpret_tensor(buf158, (768, 768), (768, 1), 0), reinterpret_tensor(buf159, (768, ), (1, ), 0), reinterpret_tensor(buf148, (768, 768), (768, 1), 0), reinterpret_tensor(buf149, (768, ), (1, ), 0), buf145, buf146, reinterpret_tensor(buf140, (3072, 768), (768, 1), 0), reinterpret_tensor(buf141, (3072, ), (1, ), 0), reinterpret_tensor(buf136, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf137, (768, ), (1, ), 0), buf132, buf133, reinterpret_tensor(buf126, (768, 768), (768, 1), 0), reinterpret_tensor(buf127, (768, ), (1, ), 0), reinterpret_tensor(buf122, (768, 768), (768, 1), 0), reinterpret_tensor(buf123, (768, ), (1, ), 0), reinterpret_tensor(buf119, (768, 768), (768, 1), 0), reinterpret_tensor(buf120, (768, ), (1, ), 0), reinterpret_tensor(buf109, (768, 768), (768, 1), 0), reinterpret_tensor(buf110, (768, ), (1, ), 0), buf106, buf107, reinterpret_tensor(buf101, (3072, 768), (768, 1), 0), reinterpret_tensor(buf102, (3072, ), (1, ), 0), reinterpret_tensor(buf97, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf98, (768, ), (1, ), 0), buf93, buf94, reinterpret_tensor(buf87, (768, 768), (768, 1), 0), reinterpret_tensor(buf88, (768, ), (1, ), 0), reinterpret_tensor(buf83, (768, 768), (768, 1), 0), reinterpret_tensor(buf84, (768, ), (1, ), 0), reinterpret_tensor(buf80, (768, 768), (768, 1), 0), reinterpret_tensor(buf81, (768, ), (1, ), 0), reinterpret_tensor(buf70, (768, 768), (768, 1), 0), reinterpret_tensor(buf71, (768, ), (1, ), 0), buf67, buf68, reinterpret_tensor(buf62, (3072, 768), (768, 1), 0), reinterpret_tensor(buf63, (3072, ), (1, ), 0), reinterpret_tensor(buf58, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf59, (768, ), (1, ), 0), buf54, buf55, reinterpret_tensor(buf48, (768, 768), (768, 1), 0), reinterpret_tensor(buf49, (768, ), (1, ), 0), reinterpret_tensor(buf44, (768, 768), (768, 1), 0), reinterpret_tensor(buf45, (768, ), (1, ), 0), reinterpret_tensor(buf41, (768, 768), (768, 1), 0), reinterpret_tensor(buf42, (768, ), (1, ), 0), reinterpret_tensor(buf31, (768, 768), (768, 1), 0), reinterpret_tensor(buf32, (768, ), (1, ), 0), buf28, buf29, reinterpret_tensor(buf23, (3072, 768), (768, 1), 0), reinterpret_tensor(buf24, (3072, ), (1, ), 0), reinterpret_tensor(buf19, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf20, (768, ), (1, ), 0), buf15, buf16, reinterpret_tensor(buf10, (2, 768), (768, 1), 0), reinterpret_tensor(buf11, (2, ), (1, ), 0), None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    slice_2 = rand_strided((1, 128), (512, 1), device='cpu', dtype=torch.int64)
    mul = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.bool)
    view = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_12 = rand_strided((1, 1, 1, 128), (128, 128, 128, 1), device='cpu', dtype=torch.bool)
    getitem_5 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_17 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_2 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_19 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_4 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_21 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_9 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.bool)
    mul_7 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_23 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_13 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_40 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_9 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_42 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_44 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_17 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.bool)
    mul_14 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_46 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_21 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_63 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_16 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_65 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_16 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_67 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_25 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.bool)
    mul_21 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_69 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_29 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_86 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_23 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_88 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_90 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_33 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.bool)
    mul_28 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_92 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_37 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_109 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_30 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_111 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_28 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_113 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_41 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.bool)
    mul_35 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_115 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_45 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.bool)
    view_132 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_37 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    view_134 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_136 = rand_strided((128, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_49 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.bool)
    mul_42 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.float32)
    getitem_53 = rand_strided((1, 128, 768), (98304, 768, 1), device='cpu', dtype=torch.bool)
    view_138 = rand_strided((128, 768), (768, 1), device='cpu', dtype=torch.float32)
    sub_20 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.float32)
    ne = rand_strided((1, ), (1, ), device='cpu', dtype=torch.bool)
    sub_22 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.float32)
    ne_3 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.bool)
    ne_6 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.bool)
    where_10 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.int64)
    ne_8 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.bool)
    where_12 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.int64)
    permute_67 = rand_strided((2, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_18 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_71 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_75 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_19 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_79 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_84 = rand_strided((12, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_85 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_10 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_86 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_87 = rand_strided((12, 128, 64), (64, 768, 1), device='cpu', dtype=torch.float32)
    permute_90 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_95 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_100 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_21 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_104 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_108 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_22 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_112 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_117 = rand_strided((12, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_118 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_11 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_119 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_120 = rand_strided((12, 128, 64), (64, 768, 1), device='cpu', dtype=torch.float32)
    permute_123 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_128 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_133 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_137 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_141 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_25 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_145 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_150 = rand_strided((12, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_151 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_12 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_152 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_153 = rand_strided((12, 128, 64), (64, 768, 1), device='cpu', dtype=torch.float32)
    permute_156 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_161 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_166 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_27 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_170 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_174 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_28 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_178 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_183 = rand_strided((12, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_184 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_13 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_185 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_186 = rand_strided((12, 128, 64), (64, 768, 1), device='cpu', dtype=torch.float32)
    permute_189 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_194 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_199 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_30 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_203 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_207 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_31 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_211 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_216 = rand_strided((12, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_217 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_14 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_218 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_219 = rand_strided((12, 128, 64), (64, 768, 1), device='cpu', dtype=torch.float32)
    permute_222 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_227 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_232 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_33 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_236 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_240 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_34 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    permute_244 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_249 = rand_strided((12, 128, 128), (16384, 1, 128), device='cpu', dtype=torch.float32)
    permute_250 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    alias_15 = rand_strided((1, 12, 128, 128), (196608, 16384, 128, 1), device='cpu', dtype=torch.float32)
    permute_251 = rand_strided((12, 64, 128), (64, 1, 768), device='cpu', dtype=torch.float32)
    permute_252 = rand_strided((12, 128, 64), (64, 768, 1), device='cpu', dtype=torch.float32)
    permute_255 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_260 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_265 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_36 = rand_strided((1, 128, 1), (128, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.float32)
    tangents_3 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_3, primals_13, primals_19, primals_29, primals_35, primals_45, primals_51, primals_61, primals_67, primals_77, primals_83, primals_93, primals_99, primals_104, slice_2, mul, getitem_3, view, view_12, getitem_5, view_17, mul_2, view_19, addmm_4, view_21, getitem_9, mul_7, view_23, getitem_13, view_40, mul_9, view_42, addmm_10, view_44, getitem_17, mul_14, view_46, getitem_21, view_63, mul_16, view_65, addmm_16, view_67, getitem_25, mul_21, view_69, getitem_29, view_86, mul_23, view_88, addmm_22, view_90, getitem_33, mul_28, view_92, getitem_37, view_109, mul_30, view_111, addmm_28, view_113, getitem_41, mul_35, view_115, getitem_45, view_132, mul_37, view_134, addmm_34, view_136, getitem_49, mul_42, getitem_53, view_138, sub_20, ne, sub_22, ne_3, ne_6, where_10, ne_8, where_12, permute_67, div_18, permute_71, permute_75, div_19, permute_79, permute_84, permute_85, alias_10, permute_86, permute_87, permute_90, permute_95, permute_100, div_21, permute_104, permute_108, div_22, permute_112, permute_117, permute_118, alias_11, permute_119, permute_120, permute_123, permute_128, permute_133, div_24, permute_137, permute_141, div_25, permute_145, permute_150, permute_151, alias_12, permute_152, permute_153, permute_156, permute_161, permute_166, div_27, permute_170, permute_174, div_28, permute_178, permute_183, permute_184, alias_13, permute_185, permute_186, permute_189, permute_194, permute_199, div_30, permute_203, permute_207, div_31, permute_211, permute_216, permute_217, alias_14, permute_218, permute_219, permute_222, permute_227, permute_232, div_33, permute_236, permute_240, div_34, permute_244, permute_249, permute_250, alias_15, permute_251, permute_252, permute_255, permute_260, permute_265, div_36, tangents_1, tangents_2, tangents_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DistilBertForQuestionAnswering', benchmark_compiled_module)
