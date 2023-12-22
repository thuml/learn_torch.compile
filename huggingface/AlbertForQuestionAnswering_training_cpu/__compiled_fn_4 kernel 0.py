
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


cpp_fused_native_layer_norm_backward_sum_3 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp11 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = static_cast<float>(4096.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 - tmp13;
                    auto tmp15 = at::vec::Vectorized<float>(tmp0);
                    auto tmp16 = tmp15 * tmp14;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (4096L*x1)));
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


cpp_fused_add_mul_pow_tanh_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
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


cpp_fused_add_native_layer_norm_backward_sum_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16384L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(4096.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_7 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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


cpp_fused_view_9 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (4096L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp9 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(4096.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16384L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(4096.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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


cpp_fused_view_19 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (4096L*x1)));
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


cpp_fused_sum_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_23 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp9 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(4096.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
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


cpp_fused_add_native_layer_norm_backward_sum_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16384L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(4096.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_27 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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


cpp_fused_view_29 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (4096L*x1)));
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


cpp_fused_sum_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
    }
}
''')


cpp_fused_sum_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_33 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp9 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(4096.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
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


cpp_fused_add_native_layer_norm_backward_sum_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16384L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(4096.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_37 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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


cpp_fused_view_39 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (4096L*x1)));
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


cpp_fused_sum_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_43 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp9 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(4096.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
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


cpp_fused_add_native_layer_norm_backward_sum_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16384L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(4096.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_47 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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


cpp_fused_view_49 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (4096L*x1)));
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


cpp_fused_sum_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_53 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp9 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(4096.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_55 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
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


cpp_fused_add_native_layer_norm_backward_sum_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16384L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(4096.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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


cpp_fused_view_59 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_60 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (4096L*x1)));
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


cpp_fused_sum_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_63 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp9 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(4096.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
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


cpp_fused_add_native_layer_norm_backward_sum_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16384L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(4096.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_67 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_68 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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


cpp_fused_view_69 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (4096L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp9 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(4096.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16384L*x1)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp13 = out_ptr2[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(4096.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_77 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_78 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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


cpp_fused_view_79 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_80 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (4096L*x1)));
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


cpp_fused_sum_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
    }
}
''')


cpp_fused_sum_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_83 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr1[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp9 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(4096.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_85 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp13 = out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(4096.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_86 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_87 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_89 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (4096L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr3[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp9 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(4096.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_91 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp13 = out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(4096.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_92 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_93 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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


cpp_fused_view_94 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_95 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (4096L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr3[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp9 = out_ptr4[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(4096.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_97 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp13 = out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(4096.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_98 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_99 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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


cpp_fused_view_100 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_101 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (4096L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_102 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (4096L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0 + (4096L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr14[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr9[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp9 = out_ptr10[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(4096.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = at::vec::Vectorized<float>(tmp0);
                    auto tmp14 = tmp13 * tmp12;
                    tmp14.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_backward_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8388608L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_104 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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


cpp_fused_add_105 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(67108864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_sum_106 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (16384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (16384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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


cpp_fused_add_107 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(67108864L); x0+=static_cast<long>(8L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_108 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp13 = out_ptr1[static_cast<long>(x0)];
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 * tmp4;
                    auto tmp6 = static_cast<float>(4096.0);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp11 - tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp0);
                    auto tmp18 = tmp17 * tmp16;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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


cpp_fused_add_109 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16777216L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_110 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_sum_view_111 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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


cpp_fused_add_112 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16777216L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_113 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_div_114 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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


cpp_fused_add_sum_115 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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


cpp_fused_add_116 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16777216L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_117 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (4096L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (4096L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_sum_view_118 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0 + (4096L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
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


cpp_fused_add_119 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16777216L); x0+=static_cast<long>(8L))
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


cpp_fused_add_120 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_embedding_dense_backward_native_layer_norm_backward_nll_loss_forward_sum_121 = async_compile.cpp('''
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        auto tmp24 = in_ptr6[static_cast<long>(x0)];
                        auto tmp28 = in_ptr7[static_cast<long>(x0)];
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
                        auto tmp18 = static_cast<int>(-1);
                        auto tmp19 = tmp17 == tmp18;
                        auto tmp20 = static_cast<float>(0.0);
                        auto tmp21 = to_float_mask(tmp19);
                        auto tmp22 = at::vec::Vectorized<float>(tmp20);
                        auto tmp23 = decltype(tmp22)::blendv(tmp16, tmp22, tmp21);
                        auto tmp25 = tmp24 == tmp18;
                        auto tmp26 = to_float_mask(tmp25);
                        auto tmp27 = decltype(tmp22)::blendv(tmp16, tmp22, tmp26);
                        auto tmp29 = static_cast<int>(0);
                        auto tmp30 = tmp28 == tmp29;
                        auto tmp31 = to_float_mask(tmp30);
                        auto tmp32 = decltype(tmp22)::blendv(tmp16, tmp22, tmp31);
                        tmp23.store(out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                        tmp27.store(out_ptr5 + static_cast<long>(x1 + (128L*x0)));
                        tmp32.store(out_ptr6 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            tmp_acc1_vec = tmp_acc1_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr7 + static_cast<long>(x0));
                        tmp_acc1_vec.store(out_ptr8 + static_cast<long>(x0));
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


cpp_fused_embedding_dense_backward_122 = async_compile.cpp('''
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


cpp_fused_embedding_dense_backward_123 = async_compile.cpp('''
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
    primals_4, primals_16, primals_22, primals_28, expand, slice_2, mul_1, view, view_2, view_18, mul_3, view_20, addmm_5, tanh, view_22, mul_9, view_24, view_40, mul_11, view_42, addmm_11, tanh_1, view_44, mul_17, view_46, view_62, mul_19, view_64, addmm_17, tanh_2, view_66, mul_25, view_68, view_84, mul_27, view_86, addmm_23, tanh_3, view_88, mul_33, view_90, view_106, mul_35, view_108, addmm_29, tanh_4, view_110, mul_41, view_112, view_128, mul_43, view_130, addmm_35, tanh_5, view_132, mul_49, view_134, view_150, mul_51, view_152, addmm_41, tanh_6, view_154, mul_57, view_156, view_172, mul_59, view_174, addmm_47, tanh_7, view_176, mul_65, view_178, view_194, mul_67, view_196, addmm_53, tanh_8, view_198, mul_73, view_200, view_216, mul_75, view_218, addmm_59, tanh_9, view_220, mul_81, view_222, view_238, mul_83, view_240, addmm_65, tanh_10, view_242, mul_89, view_244, view_260, mul_91, view_262, addmm_71, tanh_11, view_264, mul_97, view_266, sub_39, ne, sub_41, ne_3, ne_6, where_4, ne_8, where_6, permute_134, div_30, permute_138, permute_142, div_31, permute_146, permute_151, permute_152, alias_29, permute_153, permute_154, permute_159, permute_163, permute_167, div_33, div_34, permute_184, permute_185, alias_31, permute_186, permute_187, div_36, div_37, permute_217, permute_218, alias_33, permute_219, permute_220, div_39, div_40, permute_250, permute_251, alias_35, permute_252, permute_253, div_42, div_43, permute_283, permute_284, alias_37, permute_285, permute_286, div_45, div_46, permute_316, permute_317, alias_39, permute_318, permute_319, div_48, div_49, permute_349, permute_350, alias_41, permute_351, permute_352, div_51, div_52, permute_382, permute_383, alias_43, permute_384, permute_385, div_54, div_55, permute_415, permute_416, alias_45, permute_417, permute_418, div_57, div_58, permute_448, permute_449, alias_47, permute_450, permute_451, div_60, div_61, permute_481, permute_482, alias_49, permute_483, permute_484, div_63, div_64, permute_514, permute_515, alias_51, permute_516, permute_517, permute_534, div_66, tangents_1, tangents_2, tangents_3 = args
    args.clear()
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_16, (4096, ), (1, ))
    assert_size_stride(primals_22, (4096, ), (1, ))
    assert_size_stride(primals_28, (1, 512), (512, 1))
    assert_size_stride(expand, (1, 512), (512, 1))
    assert_size_stride(slice_2, (1, 512), (512, 1))
    assert_size_stride(mul_1, (1, 512, 128), (65536, 128, 1))
    assert_size_stride(view, (512, 128), (128, 1))
    assert_size_stride(view_2, (512, 4096), (4096, 1))
    assert_size_stride(view_18, (512, 4096), (4096, 1))
    assert_size_stride(mul_3, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_20, (512, 4096), (4096, 1))
    assert_size_stride(addmm_5, (512, 16384), (16384, 1))
    assert_size_stride(tanh, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_22, (512, 16384), (16384, 1))
    assert_size_stride(mul_9, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_24, (512, 4096), (4096, 1))
    assert_size_stride(view_40, (512, 4096), (4096, 1))
    assert_size_stride(mul_11, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_42, (512, 4096), (4096, 1))
    assert_size_stride(addmm_11, (512, 16384), (16384, 1))
    assert_size_stride(tanh_1, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_44, (512, 16384), (16384, 1))
    assert_size_stride(mul_17, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_46, (512, 4096), (4096, 1))
    assert_size_stride(view_62, (512, 4096), (4096, 1))
    assert_size_stride(mul_19, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_64, (512, 4096), (4096, 1))
    assert_size_stride(addmm_17, (512, 16384), (16384, 1))
    assert_size_stride(tanh_2, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_66, (512, 16384), (16384, 1))
    assert_size_stride(mul_25, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_68, (512, 4096), (4096, 1))
    assert_size_stride(view_84, (512, 4096), (4096, 1))
    assert_size_stride(mul_27, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_86, (512, 4096), (4096, 1))
    assert_size_stride(addmm_23, (512, 16384), (16384, 1))
    assert_size_stride(tanh_3, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_88, (512, 16384), (16384, 1))
    assert_size_stride(mul_33, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_90, (512, 4096), (4096, 1))
    assert_size_stride(view_106, (512, 4096), (4096, 1))
    assert_size_stride(mul_35, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_108, (512, 4096), (4096, 1))
    assert_size_stride(addmm_29, (512, 16384), (16384, 1))
    assert_size_stride(tanh_4, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_110, (512, 16384), (16384, 1))
    assert_size_stride(mul_41, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_112, (512, 4096), (4096, 1))
    assert_size_stride(view_128, (512, 4096), (4096, 1))
    assert_size_stride(mul_43, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_130, (512, 4096), (4096, 1))
    assert_size_stride(addmm_35, (512, 16384), (16384, 1))
    assert_size_stride(tanh_5, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_132, (512, 16384), (16384, 1))
    assert_size_stride(mul_49, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_134, (512, 4096), (4096, 1))
    assert_size_stride(view_150, (512, 4096), (4096, 1))
    assert_size_stride(mul_51, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_152, (512, 4096), (4096, 1))
    assert_size_stride(addmm_41, (512, 16384), (16384, 1))
    assert_size_stride(tanh_6, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_154, (512, 16384), (16384, 1))
    assert_size_stride(mul_57, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_156, (512, 4096), (4096, 1))
    assert_size_stride(view_172, (512, 4096), (4096, 1))
    assert_size_stride(mul_59, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_174, (512, 4096), (4096, 1))
    assert_size_stride(addmm_47, (512, 16384), (16384, 1))
    assert_size_stride(tanh_7, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_176, (512, 16384), (16384, 1))
    assert_size_stride(mul_65, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_178, (512, 4096), (4096, 1))
    assert_size_stride(view_194, (512, 4096), (4096, 1))
    assert_size_stride(mul_67, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_196, (512, 4096), (4096, 1))
    assert_size_stride(addmm_53, (512, 16384), (16384, 1))
    assert_size_stride(tanh_8, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_198, (512, 16384), (16384, 1))
    assert_size_stride(mul_73, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_200, (512, 4096), (4096, 1))
    assert_size_stride(view_216, (512, 4096), (4096, 1))
    assert_size_stride(mul_75, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_218, (512, 4096), (4096, 1))
    assert_size_stride(addmm_59, (512, 16384), (16384, 1))
    assert_size_stride(tanh_9, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_220, (512, 16384), (16384, 1))
    assert_size_stride(mul_81, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_222, (512, 4096), (4096, 1))
    assert_size_stride(view_238, (512, 4096), (4096, 1))
    assert_size_stride(mul_83, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_240, (512, 4096), (4096, 1))
    assert_size_stride(addmm_65, (512, 16384), (16384, 1))
    assert_size_stride(tanh_10, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_242, (512, 16384), (16384, 1))
    assert_size_stride(mul_89, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_244, (512, 4096), (4096, 1))
    assert_size_stride(view_260, (512, 4096), (4096, 1))
    assert_size_stride(mul_91, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_262, (512, 4096), (4096, 1))
    assert_size_stride(addmm_71, (512, 16384), (16384, 1))
    assert_size_stride(tanh_11, (1, 512, 16384), (8388608, 16384, 1))
    assert_size_stride(view_264, (512, 16384), (16384, 1))
    assert_size_stride(mul_97, (1, 512, 4096), (2097152, 4096, 1))
    assert_size_stride(view_266, (512, 4096), (4096, 1))
    assert_size_stride(sub_39, (1, 512), (512, 1))
    assert_size_stride(ne, (1, ), (1, ))
    assert_size_stride(sub_41, (1, 512), (512, 1))
    assert_size_stride(ne_3, (1, ), (1, ))
    assert_size_stride(ne_6, (1, 1), (1, 1))
    assert_size_stride(where_4, (1, 1), (1, 1))
    assert_size_stride(ne_8, (1, 1), (1, 1))
    assert_size_stride(where_6, (1, 1), (1, 1))
    assert_size_stride(permute_134, (2, 4096), (4096, 1))
    assert_size_stride(div_30, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_138, (4096, 16384), (16384, 1))
    assert_size_stride(permute_142, (16384, 4096), (4096, 1))
    assert_size_stride(div_31, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_146, (4096, 4096), (4096, 1))
    assert_size_stride(permute_151, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_152, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_29, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_153, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_154, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(permute_159, (4096, 4096), (4096, 1))
    assert_size_stride(permute_163, (4096, 4096), (4096, 1))
    assert_size_stride(permute_167, (4096, 4096), (4096, 1))
    assert_size_stride(div_33, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_34, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_184, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_185, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_31, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_186, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_187, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_36, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_37, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_217, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_218, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_33, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_219, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_220, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_39, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_40, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_250, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_251, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_35, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_252, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_253, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_42, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_43, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_283, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_284, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_37, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_285, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_286, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_45, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_46, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_316, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_317, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_39, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_318, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_319, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_48, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_49, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_349, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_350, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_41, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_351, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_352, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_51, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_52, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_382, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_383, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_43, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_384, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_385, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_54, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_55, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_415, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_416, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_45, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_417, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_418, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_57, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_58, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_448, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_449, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_47, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_450, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_451, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_60, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_61, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_481, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_482, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_49, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_483, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_484, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(div_63, (1, 512, 1), (512, 1, 1))
    assert_size_stride(div_64, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_514, (64, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_515, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(alias_51, (1, 64, 512, 512), (16777216, 262144, 512, 1))
    assert_size_stride(permute_516, (64, 64, 512), (64, 1, 4096))
    assert_size_stride(permute_517, (64, 512, 64), (64, 4096, 1))
    assert_size_stride(permute_534, (4096, 128), (128, 1))
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
    buf9 = empty((512, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf8, (512, 2), (2, 1), 0), permute_134, out=buf9)
    del permute_134
    buf10 = empty((2, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf8, (2, 512), (1, 2), 0), view_266, out=buf10)
    del view_266
    buf11 = empty((1, 2), device='cpu', dtype=torch.float32)
    buf12 = reinterpret_tensor(buf4, (1, 512, 1), (512, 1, 512), 0); del buf4  # reuse
    buf13 = reinterpret_tensor(buf0, (1, 512, 1), (512, 1, 512), 0); del buf0  # reuse
    buf14 = empty((1, 512, 4096), device='cpu', dtype=torch.float32)
    buf15 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf16 = empty((4096, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_backward_sum_3(c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_97.data_ptr()), c_void_p(div_30.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()))
    del buf8
    del div_30
    del mul_97
    buf17 = empty((512, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf14, (512, 4096), (4096, 1), 0), permute_138, out=buf17)
    buf18 = empty((4096, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf14, (4096, 512), (1, 4096), 0), view_264, out=buf18)
    del view_264
    buf20 = reinterpret_tensor(buf17, (1, 512, 16384), (8388608, 16384, 1), 0); del buf17  # reuse
    cpp_fused_add_mul_pow_tanh_backward_4(c_void_p(buf20.data_ptr()), c_void_p(addmm_71.data_ptr()), c_void_p(tanh_11.data_ptr()))
    del addmm_71
    del tanh_11
    buf21 = buf9; del buf9  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf20, (512, 16384), (16384, 1), 0), permute_142, out=buf21)
    buf19 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf27 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf28 = empty((4096, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_5(c_void_p(buf14.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(mul_91.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()))
    buf22 = empty((16384, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf20, (16384, 512), (1, 16384), 0), view_262, out=buf22)
    del view_262
    buf23 = empty((1, 16384), device='cpu', dtype=torch.float32)
    buf24 = buf13; del buf13  # reuse
    buf25 = buf12; del buf12  # reuse
    buf26 = buf14; del buf14  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_6(c_void_p(buf26.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_91.data_ptr()), c_void_p(div_31.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()))
    del div_31
    del mul_91
    buf29 = buf21; del buf21  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf26, (512, 4096), (4096, 1), 0), permute_146, out=buf29)
    buf30 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf26, (4096, 512), (1, 4096), 0), view_260, out=buf30)
    del view_260
    buf32 = empty((64, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_151, reinterpret_tensor(buf29, (64, 512, 64), (64, 4096, 1), 0), out=buf32)
    del permute_151
    buf38 = empty((512, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_view_7(c_void_p(buf32.data_ptr()), c_void_p(buf38.data_ptr()))
    buf39 = reinterpret_tensor(buf32, (512, 4096), (4096, 1), 0); del buf32  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf38, permute_159, out=buf39)
    buf33 = empty((64, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf29, (64, 512, 64), (64, 4096, 1), 0), permute_152, out=buf33)
    del permute_152
    buf34 = empty_strided((1, 64, 512, 1), (32768, 512, 1, 32768), device='cpu', dtype=torch.float32)
    buf35 = reinterpret_tensor(buf33, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf33  # reuse
    cpp_fused__softmax_backward_data_div_8(c_void_p(buf35.data_ptr()), c_void_p(alias_29.data_ptr()), c_void_p(buf34.data_ptr()))
    del alias_29
    buf36 = reinterpret_tensor(buf29, (64, 64, 512), (32768, 512, 1), 0); del buf29  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_153, reinterpret_tensor(buf35, (64, 512, 512), (262144, 512, 1), 0), out=buf36)
    del permute_153
    buf42 = empty((512, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf36, (512, 4096), (1, 512), 0), permute_163, out=buf42)
    buf37 = empty((64, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf35, (64, 512, 512), (262144, 512, 1), 0), permute_154, out=buf37)
    del permute_154
    buf45 = empty((512, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_view_9(c_void_p(buf37.data_ptr()), c_void_p(buf45.data_ptr()))
    buf46 = reinterpret_tensor(buf37, (512, 4096), (4096, 1), 0); del buf37  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf45, permute_167, out=buf46)
    buf31 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf53 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf54 = empty((4096, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_10(c_void_p(buf26.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(mul_89.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()))
    buf40 = reinterpret_tensor(buf35, (4096, 4096), (4096, 1), 0); del buf35  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf38, (4096, 512), (1, 4096), 0), view_244, out=buf40)
    buf41 = empty((1, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_sum_11(c_void_p(buf38.data_ptr()), c_void_p(buf41.data_ptr()))
    buf43 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf36, (4096, 512), (512, 1), 0), view_244, out=buf43)
    buf44 = empty((1, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_sum_12(c_void_p(buf36.data_ptr()), c_void_p(buf44.data_ptr()))
    buf47 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf45, (4096, 512), (1, 4096), 0), view_244, out=buf47)
    del view_244
    buf48 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf49 = buf26; del buf26  # reuse
    buf50 = buf25; del buf25  # reuse
    buf51 = buf24; del buf24  # reuse
    buf52 = buf49; del buf49  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_13(c_void_p(buf52.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_89.data_ptr()), c_void_p(div_33.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()))
    del div_33
    del mul_89
    buf55 = reinterpret_tensor(buf20, (512, 16384), (16384, 1), 0); del buf20  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf52, (512, 4096), (4096, 1), 0), permute_138, out=buf55)
    buf56 = empty((4096, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf52, (4096, 512), (1, 4096), 0), view_242, out=buf56)
    del view_242
    buf58 = reinterpret_tensor(buf55, (1, 512, 16384), (8388608, 16384, 1), 0); del buf55  # reuse
    cpp_fused_add_mul_pow_tanh_backward_14(c_void_p(buf58.data_ptr()), c_void_p(addmm_65.data_ptr()), c_void_p(tanh_10.data_ptr()))
    del addmm_65
    del tanh_10
    buf59 = buf46; del buf46  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf58, (512, 16384), (16384, 1), 0), permute_142, out=buf59)
    buf57 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf65 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf66 = empty((4096, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_15(c_void_p(buf52.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(mul_83.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()))
    buf60 = empty((16384, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf58, (16384, 512), (1, 16384), 0), view_240, out=buf60)
    del view_240
    buf61 = empty((1, 16384), device='cpu', dtype=torch.float32)
    buf62 = buf51; del buf51  # reuse
    buf63 = buf50; del buf50  # reuse
    buf64 = buf52; del buf52  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_16(c_void_p(buf64.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_83.data_ptr()), c_void_p(div_34.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()))
    del div_34
    del mul_83
    buf67 = buf59; del buf59  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf64, (512, 4096), (4096, 1), 0), permute_146, out=buf67)
    buf68 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf64, (4096, 512), (1, 4096), 0), view_238, out=buf68)
    del view_238
    buf70 = reinterpret_tensor(buf45, (64, 512, 64), (32768, 64, 1), 0); del buf45  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_184, reinterpret_tensor(buf67, (64, 512, 64), (64, 4096, 1), 0), out=buf70)
    del permute_184
    buf76 = buf42; del buf42  # reuse
    cpp_fused_view_17(c_void_p(buf70.data_ptr()), c_void_p(buf76.data_ptr()))
    buf77 = reinterpret_tensor(buf70, (512, 4096), (4096, 1), 0); del buf70  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf76, permute_159, out=buf77)
    buf71 = empty((64, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf67, (64, 512, 64), (64, 4096, 1), 0), permute_185, out=buf71)
    del permute_185
    buf72 = buf34; del buf34  # reuse
    buf73 = reinterpret_tensor(buf71, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf71  # reuse
    cpp_fused__softmax_backward_data_div_18(c_void_p(buf73.data_ptr()), c_void_p(alias_31.data_ptr()), c_void_p(buf72.data_ptr()))
    del alias_31
    buf74 = reinterpret_tensor(buf67, (64, 64, 512), (32768, 512, 1), 0); del buf67  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_186, reinterpret_tensor(buf73, (64, 512, 512), (262144, 512, 1), 0), out=buf74)
    del permute_186
    buf80 = buf39; del buf39  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf74, (512, 4096), (1, 512), 0), permute_163, out=buf80)
    buf75 = reinterpret_tensor(buf36, (64, 512, 64), (32768, 64, 1), 0); del buf36  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf73, (64, 512, 512), (262144, 512, 1), 0), permute_187, out=buf75)
    del permute_187
    buf83 = buf38; del buf38  # reuse
    cpp_fused_view_19(c_void_p(buf75.data_ptr()), c_void_p(buf83.data_ptr()))
    buf84 = reinterpret_tensor(buf75, (512, 4096), (4096, 1), 0); del buf75  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf83, permute_167, out=buf84)
    buf69 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf91 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf92 = empty((4096, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_20(c_void_p(buf64.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(mul_81.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()))
    buf78 = reinterpret_tensor(buf73, (4096, 4096), (4096, 1), 0); del buf73  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf76, (4096, 512), (1, 4096), 0), view_222, out=buf78)
    buf79 = empty((1, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_sum_21(c_void_p(buf76.data_ptr()), c_void_p(buf79.data_ptr()))
    buf81 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf74, (4096, 512), (512, 1), 0), view_222, out=buf81)
    buf82 = empty((1, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_sum_22(c_void_p(buf74.data_ptr()), c_void_p(buf82.data_ptr()))
    buf85 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf83, (4096, 512), (1, 4096), 0), view_222, out=buf85)
    del view_222
    buf86 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf87 = buf64; del buf64  # reuse
    buf88 = buf63; del buf63  # reuse
    buf89 = buf62; del buf62  # reuse
    buf90 = buf87; del buf87  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_23(c_void_p(buf90.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_81.data_ptr()), c_void_p(div_36.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()))
    del div_36
    del mul_81
    buf93 = reinterpret_tensor(buf58, (512, 16384), (16384, 1), 0); del buf58  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf90, (512, 4096), (4096, 1), 0), permute_138, out=buf93)
    buf94 = empty((4096, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf90, (4096, 512), (1, 4096), 0), view_220, out=buf94)
    del view_220
    buf96 = reinterpret_tensor(buf93, (1, 512, 16384), (8388608, 16384, 1), 0); del buf93  # reuse
    cpp_fused_add_mul_pow_tanh_backward_24(c_void_p(buf96.data_ptr()), c_void_p(addmm_59.data_ptr()), c_void_p(tanh_9.data_ptr()))
    del addmm_59
    del tanh_9
    buf97 = buf84; del buf84  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf96, (512, 16384), (16384, 1), 0), permute_142, out=buf97)
    buf95 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf103 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf104 = empty((4096, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_25(c_void_p(buf90.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(mul_75.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()))
    buf98 = empty((16384, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf96, (16384, 512), (1, 16384), 0), view_218, out=buf98)
    del view_218
    buf99 = empty((1, 16384), device='cpu', dtype=torch.float32)
    buf100 = buf89; del buf89  # reuse
    buf101 = buf88; del buf88  # reuse
    buf102 = buf90; del buf90  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_26(c_void_p(buf102.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_75.data_ptr()), c_void_p(div_37.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()))
    del div_37
    del mul_75
    buf105 = buf97; del buf97  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf102, (512, 4096), (4096, 1), 0), permute_146, out=buf105)
    buf106 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf102, (4096, 512), (1, 4096), 0), view_216, out=buf106)
    del view_216
    buf108 = reinterpret_tensor(buf83, (64, 512, 64), (32768, 64, 1), 0); del buf83  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_217, reinterpret_tensor(buf105, (64, 512, 64), (64, 4096, 1), 0), out=buf108)
    del permute_217
    buf114 = buf80; del buf80  # reuse
    cpp_fused_view_27(c_void_p(buf108.data_ptr()), c_void_p(buf114.data_ptr()))
    buf115 = reinterpret_tensor(buf108, (512, 4096), (4096, 1), 0); del buf108  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf114, permute_159, out=buf115)
    buf109 = empty((64, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf105, (64, 512, 64), (64, 4096, 1), 0), permute_218, out=buf109)
    del permute_218
    buf110 = buf72; del buf72  # reuse
    buf111 = reinterpret_tensor(buf109, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf109  # reuse
    cpp_fused__softmax_backward_data_div_28(c_void_p(buf111.data_ptr()), c_void_p(alias_33.data_ptr()), c_void_p(buf110.data_ptr()))
    del alias_33
    buf112 = reinterpret_tensor(buf105, (64, 64, 512), (32768, 512, 1), 0); del buf105  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_219, reinterpret_tensor(buf111, (64, 512, 512), (262144, 512, 1), 0), out=buf112)
    del permute_219
    buf118 = buf77; del buf77  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf112, (512, 4096), (1, 512), 0), permute_163, out=buf118)
    buf113 = reinterpret_tensor(buf74, (64, 512, 64), (32768, 64, 1), 0); del buf74  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf111, (64, 512, 512), (262144, 512, 1), 0), permute_220, out=buf113)
    del permute_220
    buf121 = buf76; del buf76  # reuse
    cpp_fused_view_29(c_void_p(buf113.data_ptr()), c_void_p(buf121.data_ptr()))
    buf122 = reinterpret_tensor(buf113, (512, 4096), (4096, 1), 0); del buf113  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf121, permute_167, out=buf122)
    buf107 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf129 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf130 = empty((4096, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_30(c_void_p(buf102.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(mul_73.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()))
    buf116 = reinterpret_tensor(buf111, (4096, 4096), (4096, 1), 0); del buf111  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf114, (4096, 512), (1, 4096), 0), view_200, out=buf116)
    buf117 = empty((1, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_sum_31(c_void_p(buf114.data_ptr()), c_void_p(buf117.data_ptr()))
    buf119 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf112, (4096, 512), (512, 1), 0), view_200, out=buf119)
    buf120 = empty((1, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_sum_32(c_void_p(buf112.data_ptr()), c_void_p(buf120.data_ptr()))
    buf123 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf121, (4096, 512), (1, 4096), 0), view_200, out=buf123)
    del view_200
    buf124 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf125 = buf102; del buf102  # reuse
    buf126 = buf101; del buf101  # reuse
    buf127 = buf100; del buf100  # reuse
    buf128 = buf125; del buf125  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_33(c_void_p(buf128.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_73.data_ptr()), c_void_p(div_39.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()))
    del div_39
    del mul_73
    buf131 = reinterpret_tensor(buf96, (512, 16384), (16384, 1), 0); del buf96  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf128, (512, 4096), (4096, 1), 0), permute_138, out=buf131)
    buf132 = empty((4096, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf128, (4096, 512), (1, 4096), 0), view_198, out=buf132)
    del view_198
    buf134 = reinterpret_tensor(buf131, (1, 512, 16384), (8388608, 16384, 1), 0); del buf131  # reuse
    cpp_fused_add_mul_pow_tanh_backward_34(c_void_p(buf134.data_ptr()), c_void_p(addmm_53.data_ptr()), c_void_p(tanh_8.data_ptr()))
    del addmm_53
    del tanh_8
    buf135 = buf122; del buf122  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf134, (512, 16384), (16384, 1), 0), permute_142, out=buf135)
    buf133 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf141 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf142 = empty((4096, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_35(c_void_p(buf128.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(mul_67.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()))
    buf136 = empty((16384, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf134, (16384, 512), (1, 16384), 0), view_196, out=buf136)
    del view_196
    buf137 = empty((1, 16384), device='cpu', dtype=torch.float32)
    buf138 = buf127; del buf127  # reuse
    buf139 = buf126; del buf126  # reuse
    buf140 = buf128; del buf128  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_36(c_void_p(buf140.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_67.data_ptr()), c_void_p(div_40.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()))
    del div_40
    del mul_67
    buf143 = buf135; del buf135  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf140, (512, 4096), (4096, 1), 0), permute_146, out=buf143)
    buf144 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf140, (4096, 512), (1, 4096), 0), view_194, out=buf144)
    del view_194
    buf146 = reinterpret_tensor(buf121, (64, 512, 64), (32768, 64, 1), 0); del buf121  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_250, reinterpret_tensor(buf143, (64, 512, 64), (64, 4096, 1), 0), out=buf146)
    del permute_250
    buf152 = buf118; del buf118  # reuse
    cpp_fused_view_37(c_void_p(buf146.data_ptr()), c_void_p(buf152.data_ptr()))
    buf153 = reinterpret_tensor(buf146, (512, 4096), (4096, 1), 0); del buf146  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf152, permute_159, out=buf153)
    buf147 = empty((64, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf143, (64, 512, 64), (64, 4096, 1), 0), permute_251, out=buf147)
    del permute_251
    buf148 = buf110; del buf110  # reuse
    buf149 = reinterpret_tensor(buf147, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf147  # reuse
    cpp_fused__softmax_backward_data_div_38(c_void_p(buf149.data_ptr()), c_void_p(alias_35.data_ptr()), c_void_p(buf148.data_ptr()))
    del alias_35
    buf150 = reinterpret_tensor(buf143, (64, 64, 512), (32768, 512, 1), 0); del buf143  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_252, reinterpret_tensor(buf149, (64, 512, 512), (262144, 512, 1), 0), out=buf150)
    del permute_252
    buf156 = buf115; del buf115  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf150, (512, 4096), (1, 512), 0), permute_163, out=buf156)
    buf151 = reinterpret_tensor(buf112, (64, 512, 64), (32768, 64, 1), 0); del buf112  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf149, (64, 512, 512), (262144, 512, 1), 0), permute_253, out=buf151)
    del permute_253
    buf159 = buf114; del buf114  # reuse
    cpp_fused_view_39(c_void_p(buf151.data_ptr()), c_void_p(buf159.data_ptr()))
    buf160 = reinterpret_tensor(buf151, (512, 4096), (4096, 1), 0); del buf151  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf159, permute_167, out=buf160)
    buf145 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf167 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf168 = empty((4096, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_40(c_void_p(buf140.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(mul_65.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()))
    buf154 = reinterpret_tensor(buf149, (4096, 4096), (4096, 1), 0); del buf149  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf152, (4096, 512), (1, 4096), 0), view_178, out=buf154)
    buf155 = empty((1, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_sum_41(c_void_p(buf152.data_ptr()), c_void_p(buf155.data_ptr()))
    buf157 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf150, (4096, 512), (512, 1), 0), view_178, out=buf157)
    buf158 = empty((1, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_sum_42(c_void_p(buf150.data_ptr()), c_void_p(buf158.data_ptr()))
    buf161 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf159, (4096, 512), (1, 4096), 0), view_178, out=buf161)
    del view_178
    buf162 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf163 = buf140; del buf140  # reuse
    buf164 = buf139; del buf139  # reuse
    buf165 = buf138; del buf138  # reuse
    buf166 = buf163; del buf163  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_43(c_void_p(buf166.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_65.data_ptr()), c_void_p(div_42.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()))
    del div_42
    del mul_65
    buf169 = reinterpret_tensor(buf134, (512, 16384), (16384, 1), 0); del buf134  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf166, (512, 4096), (4096, 1), 0), permute_138, out=buf169)
    buf170 = empty((4096, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf166, (4096, 512), (1, 4096), 0), view_176, out=buf170)
    del view_176
    buf172 = reinterpret_tensor(buf169, (1, 512, 16384), (8388608, 16384, 1), 0); del buf169  # reuse
    cpp_fused_add_mul_pow_tanh_backward_44(c_void_p(buf172.data_ptr()), c_void_p(addmm_47.data_ptr()), c_void_p(tanh_7.data_ptr()))
    del addmm_47
    del tanh_7
    buf173 = buf160; del buf160  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf172, (512, 16384), (16384, 1), 0), permute_142, out=buf173)
    buf171 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf179 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf180 = empty((4096, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_45(c_void_p(buf166.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(mul_59.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()))
    buf174 = empty((16384, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf172, (16384, 512), (1, 16384), 0), view_174, out=buf174)
    del view_174
    buf175 = empty((1, 16384), device='cpu', dtype=torch.float32)
    buf176 = buf165; del buf165  # reuse
    buf177 = buf164; del buf164  # reuse
    buf178 = buf166; del buf166  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_46(c_void_p(buf178.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_59.data_ptr()), c_void_p(div_43.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()))
    del div_43
    del mul_59
    buf181 = buf173; del buf173  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf178, (512, 4096), (4096, 1), 0), permute_146, out=buf181)
    buf182 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf178, (4096, 512), (1, 4096), 0), view_172, out=buf182)
    del view_172
    buf184 = reinterpret_tensor(buf159, (64, 512, 64), (32768, 64, 1), 0); del buf159  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_283, reinterpret_tensor(buf181, (64, 512, 64), (64, 4096, 1), 0), out=buf184)
    del permute_283
    buf190 = buf156; del buf156  # reuse
    cpp_fused_view_47(c_void_p(buf184.data_ptr()), c_void_p(buf190.data_ptr()))
    buf191 = reinterpret_tensor(buf184, (512, 4096), (4096, 1), 0); del buf184  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf190, permute_159, out=buf191)
    buf185 = empty((64, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf181, (64, 512, 64), (64, 4096, 1), 0), permute_284, out=buf185)
    del permute_284
    buf186 = buf148; del buf148  # reuse
    buf187 = reinterpret_tensor(buf185, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf185  # reuse
    cpp_fused__softmax_backward_data_div_48(c_void_p(buf187.data_ptr()), c_void_p(alias_37.data_ptr()), c_void_p(buf186.data_ptr()))
    del alias_37
    buf188 = reinterpret_tensor(buf181, (64, 64, 512), (32768, 512, 1), 0); del buf181  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_285, reinterpret_tensor(buf187, (64, 512, 512), (262144, 512, 1), 0), out=buf188)
    del permute_285
    buf194 = buf153; del buf153  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf188, (512, 4096), (1, 512), 0), permute_163, out=buf194)
    buf189 = reinterpret_tensor(buf150, (64, 512, 64), (32768, 64, 1), 0); del buf150  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf187, (64, 512, 512), (262144, 512, 1), 0), permute_286, out=buf189)
    del permute_286
    buf197 = buf152; del buf152  # reuse
    cpp_fused_view_49(c_void_p(buf189.data_ptr()), c_void_p(buf197.data_ptr()))
    buf198 = reinterpret_tensor(buf189, (512, 4096), (4096, 1), 0); del buf189  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf197, permute_167, out=buf198)
    buf183 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf205 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf206 = empty((4096, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_50(c_void_p(buf178.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(mul_57.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()))
    buf192 = reinterpret_tensor(buf187, (4096, 4096), (4096, 1), 0); del buf187  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf190, (4096, 512), (1, 4096), 0), view_156, out=buf192)
    buf193 = empty((1, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_sum_51(c_void_p(buf190.data_ptr()), c_void_p(buf193.data_ptr()))
    buf195 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf188, (4096, 512), (512, 1), 0), view_156, out=buf195)
    buf196 = empty((1, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_sum_52(c_void_p(buf188.data_ptr()), c_void_p(buf196.data_ptr()))
    buf199 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf197, (4096, 512), (1, 4096), 0), view_156, out=buf199)
    del view_156
    buf200 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf201 = buf178; del buf178  # reuse
    buf202 = buf177; del buf177  # reuse
    buf203 = buf176; del buf176  # reuse
    buf204 = buf201; del buf201  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_53(c_void_p(buf204.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_57.data_ptr()), c_void_p(div_45.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()))
    del div_45
    del mul_57
    buf207 = reinterpret_tensor(buf172, (512, 16384), (16384, 1), 0); del buf172  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf204, (512, 4096), (4096, 1), 0), permute_138, out=buf207)
    buf208 = empty((4096, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf204, (4096, 512), (1, 4096), 0), view_154, out=buf208)
    del view_154
    buf210 = reinterpret_tensor(buf207, (1, 512, 16384), (8388608, 16384, 1), 0); del buf207  # reuse
    cpp_fused_add_mul_pow_tanh_backward_54(c_void_p(buf210.data_ptr()), c_void_p(addmm_41.data_ptr()), c_void_p(tanh_6.data_ptr()))
    del addmm_41
    del tanh_6
    buf211 = buf198; del buf198  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf210, (512, 16384), (16384, 1), 0), permute_142, out=buf211)
    buf209 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf217 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf218 = empty((4096, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_55(c_void_p(buf204.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(mul_51.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()))
    buf212 = empty((16384, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf210, (16384, 512), (1, 16384), 0), view_152, out=buf212)
    del view_152
    buf213 = empty((1, 16384), device='cpu', dtype=torch.float32)
    buf214 = buf203; del buf203  # reuse
    buf215 = buf202; del buf202  # reuse
    buf216 = buf204; del buf204  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_56(c_void_p(buf216.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_51.data_ptr()), c_void_p(div_46.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()))
    del div_46
    del mul_51
    buf219 = buf211; del buf211  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf216, (512, 4096), (4096, 1), 0), permute_146, out=buf219)
    buf220 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf216, (4096, 512), (1, 4096), 0), view_150, out=buf220)
    del view_150
    buf222 = reinterpret_tensor(buf197, (64, 512, 64), (32768, 64, 1), 0); del buf197  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_316, reinterpret_tensor(buf219, (64, 512, 64), (64, 4096, 1), 0), out=buf222)
    del permute_316
    buf228 = buf194; del buf194  # reuse
    cpp_fused_view_57(c_void_p(buf222.data_ptr()), c_void_p(buf228.data_ptr()))
    buf229 = reinterpret_tensor(buf222, (512, 4096), (4096, 1), 0); del buf222  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf228, permute_159, out=buf229)
    buf223 = empty((64, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf219, (64, 512, 64), (64, 4096, 1), 0), permute_317, out=buf223)
    del permute_317
    buf224 = buf186; del buf186  # reuse
    buf225 = reinterpret_tensor(buf223, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf223  # reuse
    cpp_fused__softmax_backward_data_div_58(c_void_p(buf225.data_ptr()), c_void_p(alias_39.data_ptr()), c_void_p(buf224.data_ptr()))
    del alias_39
    buf226 = reinterpret_tensor(buf219, (64, 64, 512), (32768, 512, 1), 0); del buf219  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_318, reinterpret_tensor(buf225, (64, 512, 512), (262144, 512, 1), 0), out=buf226)
    del permute_318
    buf232 = buf191; del buf191  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf226, (512, 4096), (1, 512), 0), permute_163, out=buf232)
    buf227 = reinterpret_tensor(buf188, (64, 512, 64), (32768, 64, 1), 0); del buf188  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf225, (64, 512, 512), (262144, 512, 1), 0), permute_319, out=buf227)
    del permute_319
    buf235 = buf190; del buf190  # reuse
    cpp_fused_view_59(c_void_p(buf227.data_ptr()), c_void_p(buf235.data_ptr()))
    buf236 = reinterpret_tensor(buf227, (512, 4096), (4096, 1), 0); del buf227  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf235, permute_167, out=buf236)
    buf221 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf243 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf244 = empty((4096, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_60(c_void_p(buf216.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(mul_49.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()))
    buf230 = reinterpret_tensor(buf225, (4096, 4096), (4096, 1), 0); del buf225  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf228, (4096, 512), (1, 4096), 0), view_134, out=buf230)
    buf231 = empty((1, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_sum_61(c_void_p(buf228.data_ptr()), c_void_p(buf231.data_ptr()))
    buf233 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf226, (4096, 512), (512, 1), 0), view_134, out=buf233)
    buf234 = empty((1, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_sum_62(c_void_p(buf226.data_ptr()), c_void_p(buf234.data_ptr()))
    buf237 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf235, (4096, 512), (1, 4096), 0), view_134, out=buf237)
    del view_134
    buf238 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf239 = buf216; del buf216  # reuse
    buf240 = buf215; del buf215  # reuse
    buf241 = buf214; del buf214  # reuse
    buf242 = buf239; del buf239  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_63(c_void_p(buf242.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_49.data_ptr()), c_void_p(div_48.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()))
    del div_48
    del mul_49
    buf245 = reinterpret_tensor(buf210, (512, 16384), (16384, 1), 0); del buf210  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf242, (512, 4096), (4096, 1), 0), permute_138, out=buf245)
    buf246 = empty((4096, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf242, (4096, 512), (1, 4096), 0), view_132, out=buf246)
    del view_132
    buf248 = reinterpret_tensor(buf245, (1, 512, 16384), (8388608, 16384, 1), 0); del buf245  # reuse
    cpp_fused_add_mul_pow_tanh_backward_64(c_void_p(buf248.data_ptr()), c_void_p(addmm_35.data_ptr()), c_void_p(tanh_5.data_ptr()))
    del addmm_35
    del tanh_5
    buf249 = buf236; del buf236  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf248, (512, 16384), (16384, 1), 0), permute_142, out=buf249)
    buf247 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf255 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf256 = empty((4096, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_65(c_void_p(buf242.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(mul_43.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()))
    buf250 = empty((16384, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf248, (16384, 512), (1, 16384), 0), view_130, out=buf250)
    del view_130
    buf251 = empty((1, 16384), device='cpu', dtype=torch.float32)
    buf252 = buf241; del buf241  # reuse
    buf253 = buf240; del buf240  # reuse
    buf254 = buf242; del buf242  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_66(c_void_p(buf254.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_43.data_ptr()), c_void_p(div_49.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()))
    del div_49
    del mul_43
    buf257 = buf249; del buf249  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf254, (512, 4096), (4096, 1), 0), permute_146, out=buf257)
    buf258 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf254, (4096, 512), (1, 4096), 0), view_128, out=buf258)
    del view_128
    buf260 = reinterpret_tensor(buf235, (64, 512, 64), (32768, 64, 1), 0); del buf235  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_349, reinterpret_tensor(buf257, (64, 512, 64), (64, 4096, 1), 0), out=buf260)
    del permute_349
    buf266 = buf232; del buf232  # reuse
    cpp_fused_view_67(c_void_p(buf260.data_ptr()), c_void_p(buf266.data_ptr()))
    buf267 = reinterpret_tensor(buf260, (512, 4096), (4096, 1), 0); del buf260  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf266, permute_159, out=buf267)
    buf261 = empty((64, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf257, (64, 512, 64), (64, 4096, 1), 0), permute_350, out=buf261)
    del permute_350
    buf262 = buf224; del buf224  # reuse
    buf263 = reinterpret_tensor(buf261, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf261  # reuse
    cpp_fused__softmax_backward_data_div_68(c_void_p(buf263.data_ptr()), c_void_p(alias_41.data_ptr()), c_void_p(buf262.data_ptr()))
    del alias_41
    buf264 = reinterpret_tensor(buf257, (64, 64, 512), (32768, 512, 1), 0); del buf257  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_351, reinterpret_tensor(buf263, (64, 512, 512), (262144, 512, 1), 0), out=buf264)
    del permute_351
    buf270 = buf229; del buf229  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf264, (512, 4096), (1, 512), 0), permute_163, out=buf270)
    buf265 = reinterpret_tensor(buf226, (64, 512, 64), (32768, 64, 1), 0); del buf226  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf263, (64, 512, 512), (262144, 512, 1), 0), permute_352, out=buf265)
    del permute_352
    buf273 = buf228; del buf228  # reuse
    cpp_fused_view_69(c_void_p(buf265.data_ptr()), c_void_p(buf273.data_ptr()))
    buf274 = reinterpret_tensor(buf265, (512, 4096), (4096, 1), 0); del buf265  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf273, permute_167, out=buf274)
    buf259 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf281 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf282 = empty((4096, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_70(c_void_p(buf254.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(mul_41.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()))
    buf268 = reinterpret_tensor(buf263, (4096, 4096), (4096, 1), 0); del buf263  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf266, (4096, 512), (1, 4096), 0), view_112, out=buf268)
    buf269 = empty((1, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_sum_71(c_void_p(buf266.data_ptr()), c_void_p(buf269.data_ptr()))
    buf271 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf264, (4096, 512), (512, 1), 0), view_112, out=buf271)
    buf272 = empty((1, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_sum_72(c_void_p(buf264.data_ptr()), c_void_p(buf272.data_ptr()))
    buf275 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf273, (4096, 512), (1, 4096), 0), view_112, out=buf275)
    del view_112
    buf276 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf277 = buf254; del buf254  # reuse
    buf278 = buf253; del buf253  # reuse
    buf279 = buf252; del buf252  # reuse
    buf280 = buf277; del buf277  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_73(c_void_p(buf280.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_41.data_ptr()), c_void_p(div_51.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()))
    del div_51
    del mul_41
    buf283 = reinterpret_tensor(buf248, (512, 16384), (16384, 1), 0); del buf248  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf280, (512, 4096), (4096, 1), 0), permute_138, out=buf283)
    buf284 = empty((4096, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf280, (4096, 512), (1, 4096), 0), view_110, out=buf284)
    del view_110
    buf286 = reinterpret_tensor(buf283, (1, 512, 16384), (8388608, 16384, 1), 0); del buf283  # reuse
    cpp_fused_add_mul_pow_tanh_backward_74(c_void_p(buf286.data_ptr()), c_void_p(addmm_29.data_ptr()), c_void_p(tanh_4.data_ptr()))
    del addmm_29
    del tanh_4
    buf287 = buf274; del buf274  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf286, (512, 16384), (16384, 1), 0), permute_142, out=buf287)
    buf285 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf293 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf294 = empty((4096, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_75(c_void_p(buf280.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(mul_35.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()))
    buf288 = empty((16384, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf286, (16384, 512), (1, 16384), 0), view_108, out=buf288)
    del view_108
    buf289 = empty((1, 16384), device='cpu', dtype=torch.float32)
    buf290 = buf279; del buf279  # reuse
    buf291 = buf278; del buf278  # reuse
    buf292 = buf280; del buf280  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_76(c_void_p(buf292.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_35.data_ptr()), c_void_p(div_52.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()))
    del div_52
    del mul_35
    buf295 = buf287; del buf287  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf292, (512, 4096), (4096, 1), 0), permute_146, out=buf295)
    buf296 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf292, (4096, 512), (1, 4096), 0), view_106, out=buf296)
    del view_106
    buf298 = reinterpret_tensor(buf273, (64, 512, 64), (32768, 64, 1), 0); del buf273  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_382, reinterpret_tensor(buf295, (64, 512, 64), (64, 4096, 1), 0), out=buf298)
    del permute_382
    buf304 = buf270; del buf270  # reuse
    cpp_fused_view_77(c_void_p(buf298.data_ptr()), c_void_p(buf304.data_ptr()))
    buf305 = reinterpret_tensor(buf298, (512, 4096), (4096, 1), 0); del buf298  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf304, permute_159, out=buf305)
    buf299 = empty((64, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf295, (64, 512, 64), (64, 4096, 1), 0), permute_383, out=buf299)
    del permute_383
    buf300 = buf262; del buf262  # reuse
    buf301 = reinterpret_tensor(buf299, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf299  # reuse
    cpp_fused__softmax_backward_data_div_78(c_void_p(buf301.data_ptr()), c_void_p(alias_43.data_ptr()), c_void_p(buf300.data_ptr()))
    del alias_43
    buf302 = reinterpret_tensor(buf295, (64, 64, 512), (32768, 512, 1), 0); del buf295  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_384, reinterpret_tensor(buf301, (64, 512, 512), (262144, 512, 1), 0), out=buf302)
    del permute_384
    buf308 = buf267; del buf267  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf302, (512, 4096), (1, 512), 0), permute_163, out=buf308)
    buf303 = reinterpret_tensor(buf264, (64, 512, 64), (32768, 64, 1), 0); del buf264  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf301, (64, 512, 512), (262144, 512, 1), 0), permute_385, out=buf303)
    del permute_385
    buf311 = buf266; del buf266  # reuse
    cpp_fused_view_79(c_void_p(buf303.data_ptr()), c_void_p(buf311.data_ptr()))
    buf312 = reinterpret_tensor(buf303, (512, 4096), (4096, 1), 0); del buf303  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf311, permute_167, out=buf312)
    buf297 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf319 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf320 = empty((4096, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_80(c_void_p(buf292.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(mul_33.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()))
    buf306 = reinterpret_tensor(buf301, (4096, 4096), (4096, 1), 0); del buf301  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf304, (4096, 512), (1, 4096), 0), view_90, out=buf306)
    buf307 = empty((1, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_sum_81(c_void_p(buf304.data_ptr()), c_void_p(buf307.data_ptr()))
    buf309 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf302, (4096, 512), (512, 1), 0), view_90, out=buf309)
    buf310 = empty((1, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_sum_82(c_void_p(buf302.data_ptr()), c_void_p(buf310.data_ptr()))
    buf313 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf311, (4096, 512), (1, 4096), 0), view_90, out=buf313)
    del view_90
    buf314 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf315 = buf292; del buf292  # reuse
    buf316 = buf291; del buf291  # reuse
    buf317 = buf290; del buf290  # reuse
    buf318 = buf315; del buf315  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_83(c_void_p(buf318.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_33.data_ptr()), c_void_p(div_54.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()))
    del div_54
    del mul_33
    buf323 = reinterpret_tensor(buf286, (512, 16384), (16384, 1), 0); del buf286  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf318, (512, 4096), (4096, 1), 0), permute_138, out=buf323)
    buf328 = reinterpret_tensor(buf323, (1, 512, 16384), (8388608, 16384, 1), 0); del buf323  # reuse
    cpp_fused_add_mul_pow_tanh_backward_84(c_void_p(buf328.data_ptr()), c_void_p(addmm_23.data_ptr()), c_void_p(tanh_3.data_ptr()))
    del addmm_23
    del tanh_3
    buf329 = buf312; del buf312  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf328, (512, 16384), (16384, 1), 0), permute_142, out=buf329)
    buf334 = buf317; del buf317  # reuse
    buf335 = buf316; del buf316  # reuse
    buf336 = reinterpret_tensor(buf311, (1, 512, 4096), (2097152, 4096, 1), 0); del buf311  # reuse
    cpp_fused_add_native_layer_norm_backward_85(c_void_p(buf318.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_27.data_ptr()), c_void_p(div_55.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()))
    del div_55
    buf341 = buf308; del buf308  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf336, (512, 4096), (4096, 1), 0), permute_146, out=buf341)
    buf346 = reinterpret_tensor(buf305, (64, 512, 64), (32768, 64, 1), 0); del buf305  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_415, reinterpret_tensor(buf341, (64, 512, 64), (64, 4096, 1), 0), out=buf346)
    del permute_415
    buf352 = reinterpret_tensor(buf302, (512, 4096), (4096, 1), 0); del buf302  # reuse
    cpp_fused_view_86(c_void_p(buf346.data_ptr()), c_void_p(buf352.data_ptr()))
    buf353 = reinterpret_tensor(buf346, (512, 4096), (4096, 1), 0); del buf346  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf352, permute_159, out=buf353)
    buf347 = empty((64, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf341, (64, 512, 64), (64, 4096, 1), 0), permute_416, out=buf347)
    del permute_416
    buf348 = buf300; del buf300  # reuse
    buf349 = reinterpret_tensor(buf347, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf347  # reuse
    cpp_fused__softmax_backward_data_div_87(c_void_p(buf349.data_ptr()), c_void_p(alias_45.data_ptr()), c_void_p(buf348.data_ptr()))
    del alias_45
    buf350 = reinterpret_tensor(buf341, (64, 64, 512), (32768, 512, 1), 0); del buf341  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_417, reinterpret_tensor(buf349, (64, 512, 512), (262144, 512, 1), 0), out=buf350)
    del permute_417
    buf358 = buf304; del buf304  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf350, (512, 4096), (1, 512), 0), permute_163, out=buf358)
    buf351 = empty((64, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf349, (64, 512, 512), (262144, 512, 1), 0), permute_418, out=buf351)
    del permute_418
    buf363 = empty((512, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_view_88(c_void_p(buf351.data_ptr()), c_void_p(buf363.data_ptr()))
    buf364 = reinterpret_tensor(buf351, (512, 4096), (4096, 1), 0); del buf351  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf363, permute_167, out=buf364)
    buf343 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf373 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf374 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf369 = reinterpret_tensor(buf353, (1, 512, 4096), (2097152, 4096, 1), 0); del buf353  # reuse
    buf370 = buf335; del buf335  # reuse
    buf371 = buf334; del buf334  # reuse
    buf372 = buf369; del buf369  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_89(c_void_p(buf372.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(mul_25.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(div_57.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()))
    del div_57
    del mul_25
    buf375 = empty((512, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf372, (512, 4096), (4096, 1), 0), permute_138, out=buf375)
    buf378 = reinterpret_tensor(buf375, (1, 512, 16384), (8388608, 16384, 1), 0); del buf375  # reuse
    cpp_fused_add_mul_pow_tanh_backward_90(c_void_p(buf378.data_ptr()), c_void_p(addmm_17.data_ptr()), c_void_p(tanh_2.data_ptr()))
    del addmm_17
    del tanh_2
    buf379 = buf364; del buf364  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf378, (512, 16384), (16384, 1), 0), permute_142, out=buf379)
    buf382 = buf371; del buf371  # reuse
    buf383 = buf370; del buf370  # reuse
    buf384 = reinterpret_tensor(buf358, (1, 512, 4096), (2097152, 4096, 1), 0); del buf358  # reuse
    cpp_fused_add_native_layer_norm_backward_91(c_void_p(buf372.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_19.data_ptr()), c_void_p(div_58.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf384.data_ptr()))
    del div_58
    buf387 = empty((512, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf384, (512, 4096), (4096, 1), 0), permute_146, out=buf387)
    buf390 = empty((64, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_448, reinterpret_tensor(buf387, (64, 512, 64), (64, 4096, 1), 0), out=buf390)
    del permute_448
    buf396 = empty((512, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_view_92(c_void_p(buf390.data_ptr()), c_void_p(buf396.data_ptr()))
    buf397 = reinterpret_tensor(buf390, (512, 4096), (4096, 1), 0); del buf390  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf396, permute_159, out=buf397)
    buf391 = reinterpret_tensor(buf349, (64, 512, 512), (262144, 512, 1), 0); del buf349  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf387, (64, 512, 64), (64, 4096, 1), 0), permute_449, out=buf391)
    del permute_449
    buf392 = buf348; del buf348  # reuse
    buf393 = reinterpret_tensor(buf391, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf391  # reuse
    cpp_fused__softmax_backward_data_div_93(c_void_p(buf393.data_ptr()), c_void_p(alias_47.data_ptr()), c_void_p(buf392.data_ptr()))
    del alias_47
    buf394 = reinterpret_tensor(buf387, (64, 64, 512), (32768, 512, 1), 0); del buf387  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_450, reinterpret_tensor(buf393, (64, 512, 512), (262144, 512, 1), 0), out=buf394)
    del permute_450
    buf400 = empty((512, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf394, (512, 4096), (1, 512), 0), permute_163, out=buf400)
    buf395 = empty((64, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf393, (64, 512, 512), (262144, 512, 1), 0), permute_451, out=buf395)
    del permute_451
    buf403 = empty((512, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_view_94(c_void_p(buf395.data_ptr()), c_void_p(buf403.data_ptr()))
    buf404 = reinterpret_tensor(buf395, (512, 4096), (4096, 1), 0); del buf395  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf403, permute_167, out=buf404)
    buf389 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf411 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf412 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf407 = reinterpret_tensor(buf397, (1, 512, 4096), (2097152, 4096, 1), 0); del buf397  # reuse
    buf408 = buf383; del buf383  # reuse
    buf409 = buf382; del buf382  # reuse
    buf410 = buf407; del buf407  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_95(c_void_p(buf410.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(mul_17.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(div_60.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()))
    del div_60
    del mul_17
    buf413 = empty((512, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf410, (512, 4096), (4096, 1), 0), permute_138, out=buf413)
    buf416 = reinterpret_tensor(buf413, (1, 512, 16384), (8388608, 16384, 1), 0); del buf413  # reuse
    cpp_fused_add_mul_pow_tanh_backward_96(c_void_p(buf416.data_ptr()), c_void_p(addmm_11.data_ptr()), c_void_p(tanh_1.data_ptr()))
    del addmm_11
    del tanh_1
    buf417 = buf404; del buf404  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf416, (512, 16384), (16384, 1), 0), permute_142, out=buf417)
    buf420 = buf409; del buf409  # reuse
    buf421 = buf408; del buf408  # reuse
    buf422 = reinterpret_tensor(buf400, (1, 512, 4096), (2097152, 4096, 1), 0); del buf400  # reuse
    cpp_fused_add_native_layer_norm_backward_97(c_void_p(buf410.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_11.data_ptr()), c_void_p(div_61.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf422.data_ptr()))
    del div_61
    buf425 = empty((512, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf422, (512, 4096), (4096, 1), 0), permute_146, out=buf425)
    buf428 = empty((64, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_481, reinterpret_tensor(buf425, (64, 512, 64), (64, 4096, 1), 0), out=buf428)
    del permute_481
    buf434 = empty((512, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_view_98(c_void_p(buf428.data_ptr()), c_void_p(buf434.data_ptr()))
    buf435 = reinterpret_tensor(buf428, (512, 4096), (4096, 1), 0); del buf428  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf434, permute_159, out=buf435)
    buf429 = reinterpret_tensor(buf393, (64, 512, 512), (262144, 512, 1), 0); del buf393  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf425, (64, 512, 64), (64, 4096, 1), 0), permute_482, out=buf429)
    del permute_482
    buf430 = buf392; del buf392  # reuse
    buf431 = reinterpret_tensor(buf429, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf429  # reuse
    cpp_fused__softmax_backward_data_div_99(c_void_p(buf431.data_ptr()), c_void_p(alias_49.data_ptr()), c_void_p(buf430.data_ptr()))
    del alias_49
    buf432 = reinterpret_tensor(buf425, (64, 64, 512), (32768, 512, 1), 0); del buf425  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_483, reinterpret_tensor(buf431, (64, 512, 512), (262144, 512, 1), 0), out=buf432)
    del permute_483
    buf438 = empty((512, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf432, (512, 4096), (1, 512), 0), permute_163, out=buf438)
    buf433 = empty((64, 512, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf431, (64, 512, 512), (262144, 512, 1), 0), permute_484, out=buf433)
    del permute_484
    buf441 = empty((512, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_view_100(c_void_p(buf433.data_ptr()), c_void_p(buf441.data_ptr()))
    buf442 = reinterpret_tensor(buf433, (512, 4096), (4096, 1), 0); del buf433  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf441, permute_167, out=buf442)
    buf427 = empty((1, 4096), device='cpu', dtype=torch.float32)
    buf449 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf450 = empty((4096, ), device='cpu', dtype=torch.float32)
    buf321 = buf129; del buf129  # reuse
    buf451 = buf321; del buf321  # reuse
    buf322 = buf130; del buf130  # reuse
    buf452 = buf322; del buf322  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_101(c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(mul_9.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf450.data_ptr()))
    del buf15
    del buf16
    del buf167
    del buf168
    del buf205
    del buf206
    del buf243
    del buf244
    del buf281
    del buf282
    buf324 = empty((4096, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf318, (4096, 512), (1, 4096), 0), view_88, out=buf324)
    del view_88
    buf325 = reinterpret_tensor(buf92, (1, 4096), (4096, 1), 0); del buf92  # reuse
    buf337 = buf91; del buf91  # reuse
    buf338 = buf54; del buf54  # reuse
    buf377 = reinterpret_tensor(buf53, (1, 4096), (4096, 1), 0); del buf53  # reuse
    buf385 = buf450; del buf450  # reuse
    buf386 = buf449; del buf449  # reuse
    buf415 = reinterpret_tensor(buf412, (1, 4096), (4096, 1), 0); del buf412  # reuse
    buf423 = buf411; del buf411  # reuse
    buf424 = buf374; del buf374  # reuse
    buf445 = reinterpret_tensor(buf435, (1, 512, 4096), (2097152, 4096, 1), 0); del buf435  # reuse
    buf446 = buf421; del buf421  # reuse
    buf447 = buf420; del buf420  # reuse
    buf448 = buf445; del buf445  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_102(c_void_p(buf448.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(mul_27.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(mul_19.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(mul_11.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(mul_9.data_ptr()), c_void_p(div_63.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()))
    del buf318
    del buf329
    del buf379
    del buf417
    del buf438
    del div_63
    del mul_11
    del mul_19
    del mul_27
    del mul_9
    del primals_22
    buf453 = empty((512, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf448, (512, 4096), (4096, 1), 0), permute_138, out=buf453)
    del permute_138
    buf458 = reinterpret_tensor(buf453, (1, 512, 16384), (8388608, 16384, 1), 0); del buf453  # reuse
    cpp_fused_add_mul_pow_tanh_backward_103(c_void_p(buf458.data_ptr()), c_void_p(addmm_5.data_ptr()), c_void_p(tanh.data_ptr()))
    del addmm_5
    del tanh
    buf459 = buf442; del buf442  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf458, (512, 16384), (16384, 1), 0), permute_142, out=buf459)
    del permute_142
    buf455 = reinterpret_tensor(buf373, (1, 4096), (4096, 1), 0); del buf373  # reuse
    buf467 = buf320; del buf320  # reuse
    buf468 = buf319; del buf319  # reuse
    buf326 = reinterpret_tensor(buf133, (4096, ), (1, ), 0); del buf133  # reuse
    buf456 = buf326; del buf326  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_104(c_void_p(buf456.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(mul_3.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()))
    del buf171
    del buf19
    del buf209
    del buf247
    del buf285
    del buf325
    del buf377
    del buf415
    del buf455
    del buf57
    del buf95
    buf376 = empty((4096, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf372, (4096, 512), (1, 4096), 0), view_66, out=buf376)
    del buf372
    del view_66
    buf414 = empty((4096, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf410, (4096, 512), (1, 4096), 0), view_44, out=buf414)
    del buf410
    del view_44
    buf454 = empty((4096, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf448, (4096, 512), (1, 4096), 0), view_22, out=buf454)
    del view_22
    buf327 = buf132; del buf132  # reuse
    buf457 = buf327; del buf327  # reuse
    cpp_fused_add_105(c_void_p(buf457.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf454.data_ptr()))
    del buf170
    del buf18
    del buf208
    del buf246
    del buf284
    del buf324
    del buf376
    buf330 = reinterpret_tensor(buf94, (16384, 4096), (4096, 1), 0); del buf94  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf328, (16384, 512), (1, 16384), 0), view_86, out=buf330)
    del view_86
    buf331 = empty((1, 16384), device='cpu', dtype=torch.float32)
    buf381 = empty((1, 16384), device='cpu', dtype=torch.float32)
    buf419 = empty((1, 16384), device='cpu', dtype=torch.float32)
    buf461 = empty((1, 16384), device='cpu', dtype=torch.float32)
    buf332 = reinterpret_tensor(buf137, (16384, ), (1, ), 0); del buf137  # reuse
    buf462 = buf332; del buf332  # reuse
    cpp_fused_add_sum_106(c_void_p(buf462.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf461.data_ptr()))
    del buf175
    del buf213
    del buf23
    del buf251
    del buf289
    del buf328
    del buf331
    del buf381
    del buf419
    del buf461
    del buf61
    del buf99
    buf380 = reinterpret_tensor(buf56, (16384, 4096), (4096, 1), 0); del buf56  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf378, (16384, 512), (1, 16384), 0), view_64, out=buf380)
    del buf378
    del view_64
    buf418 = reinterpret_tensor(buf454, (16384, 4096), (4096, 1), 0); del buf454  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf416, (16384, 512), (1, 16384), 0), view_42, out=buf418)
    del buf416
    del view_42
    buf460 = reinterpret_tensor(buf414, (16384, 4096), (4096, 1), 0); del buf414  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf458, (16384, 512), (1, 16384), 0), view_20, out=buf460)
    del buf458
    del view_20
    buf333 = buf136; del buf136  # reuse
    buf463 = buf333; del buf333  # reuse
    buf339 = buf103; del buf103  # reuse
    buf469 = buf339; del buf339  # reuse
    buf340 = buf104; del buf104  # reuse
    buf470 = buf340; del buf340  # reuse
    cpp_fused_add_107(c_void_p(buf463.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf468.data_ptr()))
    del buf141
    del buf142
    del buf174
    del buf179
    del buf180
    del buf212
    del buf217
    del buf218
    del buf22
    del buf250
    del buf255
    del buf256
    del buf27
    del buf28
    del buf288
    del buf293
    del buf294
    del buf330
    del buf337
    del buf338
    del buf380
    del buf385
    del buf386
    del buf418
    del buf423
    del buf424
    del buf460
    del buf467
    del buf468
    del buf60
    del buf65
    del buf98
    buf342 = reinterpret_tensor(buf431, (4096, 4096), (4096, 1), 0); del buf431  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf336, (4096, 512), (1, 4096), 0), view_84, out=buf342)
    del buf336
    del view_84
    buf464 = buf447; del buf447  # reuse
    buf465 = buf446; del buf446  # reuse
    buf466 = buf448; del buf448  # reuse
    buf473 = reinterpret_tensor(buf66, (1, 4096), (4096, 1), 0); del buf66  # reuse
    buf344 = reinterpret_tensor(buf107, (4096, ), (1, ), 0); del buf107  # reuse
    buf474 = buf344; del buf344  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_108(c_void_p(buf466.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_3.data_ptr()), c_void_p(div_64.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf473.data_ptr()))
    del buf145
    del buf183
    del buf221
    del buf259
    del buf297
    del buf31
    del buf343
    del buf459
    del div_64
    del mul_3
    del primals_16
    buf388 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf384, (4096, 512), (1, 4096), 0), view_62, out=buf388)
    del view_62
    buf426 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf422, (4096, 512), (1, 4096), 0), view_40, out=buf426)
    del view_40
    buf472 = empty((4096, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf466, (4096, 512), (1, 4096), 0), view_18, out=buf472)
    del view_18
    buf345 = buf106; del buf106  # reuse
    buf475 = buf345; del buf345  # reuse
    cpp_fused_add_109(c_void_p(buf475.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf472.data_ptr()))
    del buf144
    del buf182
    del buf220
    del buf258
    del buf296
    del buf30
    del buf342
    buf354 = buf68; del buf68  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf352, (4096, 512), (1, 4096), 0), view_68, out=buf354)
    buf355 = buf69; del buf69  # reuse
    buf399 = buf473; del buf473  # reuse
    buf437 = buf427; del buf427  # reuse
    cpp_fused_sum_110(c_void_p(buf352.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf437.data_ptr()))
    buf471 = buf352; del buf352  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf466, (512, 4096), (4096, 1), 0), permute_146, out=buf471)
    del permute_146
    buf476 = reinterpret_tensor(buf422, (64, 512, 64), (32768, 64, 1), 0); del buf422  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_514, reinterpret_tensor(buf471, (64, 512, 64), (64, 4096, 1), 0), out=buf476)
    del permute_514
    buf482 = reinterpret_tensor(buf384, (512, 4096), (4096, 1), 0); del buf384  # reuse
    buf485 = buf389; del buf389  # reuse
    buf356 = reinterpret_tensor(buf117, (4096, ), (1, ), 0); del buf117  # reuse
    buf486 = buf356; del buf356  # reuse
    cpp_fused_add_sum_view_111(c_void_p(buf486.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf485.data_ptr()))
    del buf155
    del buf193
    del buf231
    del buf269
    del buf307
    del buf355
    del buf399
    del buf476
    buf398 = buf472; del buf472  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf396, (4096, 512), (1, 4096), 0), view_46, out=buf398)
    del buf396
    buf436 = buf426; del buf426  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf434, (4096, 512), (1, 4096), 0), view_24, out=buf436)
    del buf434
    buf484 = buf388; del buf388  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf482, (4096, 512), (1, 4096), 0), view_2, out=buf484)
    buf357 = buf116; del buf116  # reuse
    buf487 = buf357; del buf357  # reuse
    cpp_fused_add_112(c_void_p(buf487.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf484.data_ptr()))
    del buf154
    del buf192
    del buf230
    del buf268
    del buf306
    del buf354
    buf359 = buf78; del buf78  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf350, (4096, 512), (512, 1), 0), view_68, out=buf359)
    buf360 = buf79; del buf79  # reuse
    buf402 = buf485; del buf485  # reuse
    buf440 = buf437; del buf437  # reuse
    cpp_fused_sum_113(c_void_p(buf350.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf440.data_ptr()))
    del buf350
    buf477 = reinterpret_tensor(buf484, (64, 512, 512), (262144, 512, 1), 0); del buf484  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf471, (64, 512, 64), (64, 4096, 1), 0), permute_515, out=buf477)
    del permute_515
    buf478 = buf430; del buf430  # reuse
    buf479 = reinterpret_tensor(buf477, (1, 64, 512, 512), (16777216, 262144, 512, 1), 0); del buf477  # reuse
    cpp_fused__softmax_backward_data_div_114(c_void_p(buf479.data_ptr()), c_void_p(alias_51.data_ptr()), c_void_p(buf478.data_ptr()))
    del alias_51
    del buf478
    buf480 = reinterpret_tensor(buf471, (64, 64, 512), (32768, 512, 1), 0); del buf471  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_516, reinterpret_tensor(buf479, (64, 512, 512), (262144, 512, 1), 0), out=buf480)
    del permute_516
    buf490 = buf41; del buf41  # reuse
    buf361 = reinterpret_tensor(buf120, (4096, ), (1, ), 0); del buf120  # reuse
    buf491 = buf361; del buf361  # reuse
    cpp_fused_add_sum_115(c_void_p(buf491.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf490.data_ptr()))
    del buf158
    del buf196
    del buf234
    del buf272
    del buf310
    del buf360
    del buf402
    buf401 = buf436; del buf436  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf394, (4096, 512), (512, 1), 0), view_46, out=buf401)
    del buf394
    buf439 = buf40; del buf40  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf432, (4096, 512), (512, 1), 0), view_24, out=buf439)
    buf489 = buf398; del buf398  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf480, (4096, 512), (512, 1), 0), view_2, out=buf489)
    buf362 = buf119; del buf119  # reuse
    buf492 = buf362; del buf362  # reuse
    cpp_fused_add_116(c_void_p(buf492.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf489.data_ptr()))
    del buf157
    del buf195
    del buf233
    del buf271
    del buf309
    del buf359
    del buf401
    del buf43
    buf365 = buf81; del buf81  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf363, (4096, 512), (1, 4096), 0), view_68, out=buf365)
    del view_68
    buf366 = buf82; del buf82  # reuse
    buf406 = buf490; del buf490  # reuse
    buf444 = buf440; del buf440  # reuse
    cpp_fused_sum_117(c_void_p(buf363.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf444.data_ptr()))
    buf481 = reinterpret_tensor(buf363, (64, 512, 64), (32768, 64, 1), 0); del buf363  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf479, (64, 512, 512), (262144, 512, 1), 0), permute_517, out=buf481)
    del permute_517
    buf493 = reinterpret_tensor(buf432, (512, 4096), (4096, 1), 0); del buf432  # reuse
    buf496 = buf44; del buf44  # reuse
    buf367 = reinterpret_tensor(buf124, (4096, ), (1, ), 0); del buf124  # reuse
    buf497 = buf367; del buf367  # reuse
    cpp_fused_add_sum_view_118(c_void_p(buf497.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf496.data_ptr()))
    del buf162
    del buf200
    del buf238
    del buf276
    del buf314
    del buf366
    del buf406
    del buf444
    del buf48
    del buf481
    del buf496
    buf405 = reinterpret_tensor(buf479, (4096, 4096), (4096, 1), 0); del buf479  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf403, (4096, 512), (1, 4096), 0), view_46, out=buf405)
    del buf403
    del view_46
    buf443 = buf489; del buf489  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf441, (4096, 512), (1, 4096), 0), view_24, out=buf443)
    del view_24
    buf495 = buf439; del buf439  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf493, (4096, 512), (1, 4096), 0), view_2, out=buf495)
    del view_2
    buf368 = buf123; del buf123  # reuse
    buf498 = buf368; del buf368  # reuse
    cpp_fused_add_119(c_void_p(buf498.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf495.data_ptr()))
    del buf161
    del buf199
    del buf237
    del buf275
    del buf313
    del buf365
    del buf405
    del buf443
    del buf47
    del buf495
    del buf85
    buf483 = buf441; del buf441  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf482, permute_159, out=buf483)
    del permute_159
    buf488 = buf482; del buf482  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf480, (512, 4096), (1, 512), 0), permute_163, out=buf488)
    del permute_163
    buf494 = reinterpret_tensor(buf480, (512, 4096), (4096, 1), 0); del buf480  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf493, permute_167, out=buf494)
    del buf493
    del permute_167
    buf499 = buf466; del buf466  # reuse
    cpp_fused_add_120(c_void_p(buf499.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf494.data_ptr()))
    del buf483
    del buf488
    del buf494
    buf500 = empty((512, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf499, (512, 4096), (4096, 1), 0), permute_534, out=buf500)
    del permute_534
    buf501 = empty((4096, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf499, (4096, 512), (1, 4096), 0), view, out=buf501)
    del view
    buf502 = buf86; del buf86  # reuse
    buf503 = buf465; del buf465  # reuse
    buf504 = buf464; del buf464  # reuse
    buf509 = empty((1, 512, 128), device='cpu', dtype=torch.float32)
    buf513 = empty((1, 512, 128), device='cpu', dtype=torch.float32)
    buf517 = empty((1, 512, 128), device='cpu', dtype=torch.float32)
    buf506 = empty((128, ), device='cpu', dtype=torch.float32)
    buf507 = empty((128, ), device='cpu', dtype=torch.float32)
    buf508 = empty((512, 128), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_native_layer_norm_backward_nll_loss_forward_sum_121(c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(mul_1.data_ptr()), c_void_p(div_66.data_ptr()), c_void_p(slice_2.data_ptr()), c_void_p(expand.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf508.data_ptr()))
    del buf499
    del buf500
    del buf503
    del buf504
    del div_66
    del mul_1
    del primals_4
    aten.index_put_(buf508, [slice_2], buf509, True)
    del buf509
    del slice_2
    buf512 = empty((2, 128), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_122(c_void_p(buf512.data_ptr()))
    aten.index_put_(buf512, [expand], buf513, True)
    del buf513
    del expand
    buf516 = empty((30000, 128), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_123(c_void_p(buf516.data_ptr()))
    aten.index_put_(buf516, [primals_28], buf517, True)
    del buf517
    del primals_28
    return (buf516, buf512, buf508, buf506, buf507, reinterpret_tensor(buf501, (4096, 128), (128, 1), 0), reinterpret_tensor(buf502, (4096, ), (1, ), 0), buf498, buf497, buf492, buf491, buf487, buf486, buf475, buf474, buf469, buf470, buf463, buf462, buf457, buf456, buf451, buf452, reinterpret_tensor(buf10, (2, 4096), (4096, 1), 0), reinterpret_tensor(buf11, (2, ), (1, ), 0), None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    expand = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    slice_2 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    mul_1 = rand_strided((1, 512, 128), (65536, 128, 1), device='cpu', dtype=torch.float32)
    view = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    view_2 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_18 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    mul_3 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_20 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    addmm_5 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    tanh = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cpu', dtype=torch.float32)
    view_22 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    mul_9 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_24 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_40 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    mul_11 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_42 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    addmm_11 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    tanh_1 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cpu', dtype=torch.float32)
    view_44 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    mul_17 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_46 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_62 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    mul_19 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_64 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    addmm_17 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    tanh_2 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cpu', dtype=torch.float32)
    view_66 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    mul_25 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_68 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_84 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    mul_27 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_86 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    addmm_23 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    tanh_3 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cpu', dtype=torch.float32)
    view_88 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    mul_33 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_90 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_106 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    mul_35 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_108 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    addmm_29 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    tanh_4 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cpu', dtype=torch.float32)
    view_110 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    mul_41 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_112 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_128 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    mul_43 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_130 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    addmm_35 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    tanh_5 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cpu', dtype=torch.float32)
    view_132 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    mul_49 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_134 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_150 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    mul_51 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_152 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    addmm_41 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    tanh_6 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cpu', dtype=torch.float32)
    view_154 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    mul_57 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_156 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_172 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    mul_59 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_174 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    addmm_47 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    tanh_7 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cpu', dtype=torch.float32)
    view_176 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    mul_65 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_178 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_194 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    mul_67 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_196 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    addmm_53 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    tanh_8 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cpu', dtype=torch.float32)
    view_198 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    mul_73 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_200 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_216 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    mul_75 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_218 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    addmm_59 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    tanh_9 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cpu', dtype=torch.float32)
    view_220 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    mul_81 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_222 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_238 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    mul_83 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_240 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    addmm_65 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    tanh_10 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cpu', dtype=torch.float32)
    view_242 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    mul_89 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_244 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    view_260 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    mul_91 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_262 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    addmm_71 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    tanh_11 = rand_strided((1, 512, 16384), (8388608, 16384, 1), device='cpu', dtype=torch.float32)
    view_264 = rand_strided((512, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    mul_97 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cpu', dtype=torch.float32)
    view_266 = rand_strided((512, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    sub_39 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    ne = rand_strided((1, ), (1, ), device='cpu', dtype=torch.bool)
    sub_41 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    ne_3 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.bool)
    ne_6 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.bool)
    where_4 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.int64)
    ne_8 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.bool)
    where_6 = rand_strided((1, 1), (1, 1), device='cpu', dtype=torch.int64)
    permute_134 = rand_strided((2, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    div_30 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_138 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    permute_142 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    div_31 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_146 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_151 = rand_strided((64, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_152 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    alias_29 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_153 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    permute_154 = rand_strided((64, 512, 64), (64, 4096, 1), device='cpu', dtype=torch.float32)
    permute_159 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_163 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    permute_167 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    div_33 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_34 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_184 = rand_strided((64, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_185 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    alias_31 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_186 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    permute_187 = rand_strided((64, 512, 64), (64, 4096, 1), device='cpu', dtype=torch.float32)
    div_36 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_37 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_217 = rand_strided((64, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_218 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    alias_33 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_219 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    permute_220 = rand_strided((64, 512, 64), (64, 4096, 1), device='cpu', dtype=torch.float32)
    div_39 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_40 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_250 = rand_strided((64, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_251 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    alias_35 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_252 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    permute_253 = rand_strided((64, 512, 64), (64, 4096, 1), device='cpu', dtype=torch.float32)
    div_42 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_43 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_283 = rand_strided((64, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_284 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    alias_37 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_285 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    permute_286 = rand_strided((64, 512, 64), (64, 4096, 1), device='cpu', dtype=torch.float32)
    div_45 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_46 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_316 = rand_strided((64, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_317 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    alias_39 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_318 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    permute_319 = rand_strided((64, 512, 64), (64, 4096, 1), device='cpu', dtype=torch.float32)
    div_48 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_49 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_349 = rand_strided((64, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_350 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    alias_41 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_351 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    permute_352 = rand_strided((64, 512, 64), (64, 4096, 1), device='cpu', dtype=torch.float32)
    div_51 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_52 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_382 = rand_strided((64, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_383 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    alias_43 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_384 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    permute_385 = rand_strided((64, 512, 64), (64, 4096, 1), device='cpu', dtype=torch.float32)
    div_54 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_55 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_415 = rand_strided((64, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_416 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    alias_45 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_417 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    permute_418 = rand_strided((64, 512, 64), (64, 4096, 1), device='cpu', dtype=torch.float32)
    div_57 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_58 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_448 = rand_strided((64, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_449 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    alias_47 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_450 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    permute_451 = rand_strided((64, 512, 64), (64, 4096, 1), device='cpu', dtype=torch.float32)
    div_60 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_61 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_481 = rand_strided((64, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_482 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    alias_49 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_483 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    permute_484 = rand_strided((64, 512, 64), (64, 4096, 1), device='cpu', dtype=torch.float32)
    div_63 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    div_64 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_514 = rand_strided((64, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_515 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    alias_51 = rand_strided((1, 64, 512, 512), (16777216, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_516 = rand_strided((64, 64, 512), (64, 1, 4096), device='cpu', dtype=torch.float32)
    permute_517 = rand_strided((64, 512, 64), (64, 4096, 1), device='cpu', dtype=torch.float32)
    permute_534 = rand_strided((4096, 128), (128, 1), device='cpu', dtype=torch.float32)
    div_66 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    tangents_3 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_16, primals_22, primals_28, expand, slice_2, mul_1, view, view_2, view_18, mul_3, view_20, addmm_5, tanh, view_22, mul_9, view_24, view_40, mul_11, view_42, addmm_11, tanh_1, view_44, mul_17, view_46, view_62, mul_19, view_64, addmm_17, tanh_2, view_66, mul_25, view_68, view_84, mul_27, view_86, addmm_23, tanh_3, view_88, mul_33, view_90, view_106, mul_35, view_108, addmm_29, tanh_4, view_110, mul_41, view_112, view_128, mul_43, view_130, addmm_35, tanh_5, view_132, mul_49, view_134, view_150, mul_51, view_152, addmm_41, tanh_6, view_154, mul_57, view_156, view_172, mul_59, view_174, addmm_47, tanh_7, view_176, mul_65, view_178, view_194, mul_67, view_196, addmm_53, tanh_8, view_198, mul_73, view_200, view_216, mul_75, view_218, addmm_59, tanh_9, view_220, mul_81, view_222, view_238, mul_83, view_240, addmm_65, tanh_10, view_242, mul_89, view_244, view_260, mul_91, view_262, addmm_71, tanh_11, view_264, mul_97, view_266, sub_39, ne, sub_41, ne_3, ne_6, where_4, ne_8, where_6, permute_134, div_30, permute_138, permute_142, div_31, permute_146, permute_151, permute_152, alias_29, permute_153, permute_154, permute_159, permute_163, permute_167, div_33, div_34, permute_184, permute_185, alias_31, permute_186, permute_187, div_36, div_37, permute_217, permute_218, alias_33, permute_219, permute_220, div_39, div_40, permute_250, permute_251, alias_35, permute_252, permute_253, div_42, div_43, permute_283, permute_284, alias_37, permute_285, permute_286, div_45, div_46, permute_316, permute_317, alias_39, permute_318, permute_319, div_48, div_49, permute_349, permute_350, alias_41, permute_351, permute_352, div_51, div_52, permute_382, permute_383, alias_43, permute_384, permute_385, div_54, div_55, permute_415, permute_416, alias_45, permute_417, permute_418, div_57, div_58, permute_448, permute_449, alias_47, permute_450, permute_451, div_60, div_61, permute_481, permute_482, alias_49, permute_483, permute_484, div_63, div_64, permute_514, permute_515, alias_51, permute_516, permute_517, permute_534, div_66, tangents_1, tangents_2, tangents_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AlbertForQuestionAnswering', benchmark_compiled_module)
