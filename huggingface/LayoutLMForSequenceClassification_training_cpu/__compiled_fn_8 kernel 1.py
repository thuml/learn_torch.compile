
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


cpp_fused_add_native_dropout_backward_sum_tanh_backward_view_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(x0)];
            out_ptr0[static_cast<long>(x0)] = tmp0;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp2 = in_ptr2[static_cast<long>(x0)];
            auto tmp8 = in_ptr3[static_cast<long>(x0)];
            auto tmp3 = c10::convert<float>(tmp2);
            auto tmp4 = static_cast<float>(1.1111111111111112);
            auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
            auto tmp6 = decltype(tmp1)(tmp1 * tmp5);
            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
            auto tmp9 = decltype(tmp8)(tmp8 * tmp8);
            auto tmp10 = static_cast<float>(1.0);
            auto tmp11 = decltype(tmp10)(tmp10 - tmp9);
            auto tmp12 = decltype(tmp7)(tmp7 * tmp11);
            in_out_ptr0[static_cast<long>(x0)] = tmp12;
            out_ptr1[static_cast<long>(x0)] = tmp12;
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_select_backward_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (768L*x0))];
                        auto tmp4 = in_ptr1[static_cast<long>(x1)];
                        auto tmp8 = in_ptr2[static_cast<long>(x1)];
                        auto tmp10 = in_ptr3[static_cast<long>(x1 + (768L*x0))];
                        auto tmp1 = c10::convert<int>(x0);
                        auto tmp2 = static_cast<int>(0);
                        auto tmp3 = tmp1 == tmp2;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        tmp_acc0 = tmp_acc0 + tmp9;
                        tmp_acc1 = tmp_acc1 + tmp11;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = in_ptr0[static_cast<long>(x1 + (768L*x0))];
                    auto tmp5 = in_ptr1[static_cast<long>(x1)];
                    auto tmp9 = in_ptr2[static_cast<long>(x1)];
                    auto tmp13 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = in_ptr3[static_cast<long>(x1 + (768L*x0))];
                    auto tmp16 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = c10::convert<int>(x0);
                    auto tmp3 = static_cast<int>(0);
                    auto tmp4 = tmp2 == tmp3;
                    auto tmp6 = static_cast<float>(0.0);
                    auto tmp7 = tmp4 ? tmp5 : tmp6;
                    auto tmp8 = decltype(tmp1)(tmp1 + tmp7);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                    auto tmp18 = decltype(tmp14)(tmp14 - tmp17);
                    auto tmp19 = decltype(tmp0)(tmp0 * tmp18);
                    out_ptr2[static_cast<long>(x1 + (768L*x0))] = tmp19;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (768L*x1))];
                        auto tmp4 = in_ptr1[static_cast<long>(x0)];
                        auto tmp8 = in_ptr3[static_cast<long>(x0 + (768L*x1))];
                        auto tmp1 = c10::convert<int>(x1);
                        auto tmp2 = static_cast<int>(0);
                        auto tmp3 = tmp1 == tmp2;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                        tmp_acc0 = tmp_acc0 + tmp9;
                        tmp_acc1 = tmp_acc1 + tmp7;
                    }
                    out_ptr3[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr4[static_cast<long>(x0)] = tmp_acc1;
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


cpp_fused_gelu_gelu_backward_sum_2 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_3 = async_compile.cpp('''
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


cpp_fused_sum_4 = async_compile.cpp('''
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


cpp_fused_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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


cpp_fused_view_6 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_view_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_9 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_11 = async_compile.cpp('''
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


cpp_fused_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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


cpp_fused_view_14 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_view_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_17 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_gelu_gelu_backward_sum_18 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_19 = async_compile.cpp('''
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


cpp_fused_sum_20 = async_compile.cpp('''
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


cpp_fused_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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


cpp_fused_view_22 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_view_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_25 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_27 = async_compile.cpp('''
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


cpp_fused_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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


cpp_fused_view_30 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_view_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_33 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_35 = async_compile.cpp('''
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


cpp_fused_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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


cpp_fused_view_38 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_view_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_41 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_gelu_gelu_backward_sum_42 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_43 = async_compile.cpp('''
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


cpp_fused_sum_44 = async_compile.cpp('''
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


cpp_fused_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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


cpp_fused_view_46 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_view_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_49 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_51 = async_compile.cpp('''
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


cpp_fused_sum_52 = async_compile.cpp('''
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


cpp_fused_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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


cpp_fused_view_54 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_view_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_57 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_59 = async_compile.cpp('''
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


cpp_fused_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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


cpp_fused_view_62 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_view_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_65 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_gelu_gelu_backward_sum_66 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_67 = async_compile.cpp('''
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


cpp_fused_sum_68 = async_compile.cpp('''
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


cpp_fused_69 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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


cpp_fused_view_70 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_view_72 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_73 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_gelu_gelu_backward_sum_74 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_75 = async_compile.cpp('''
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


cpp_fused_sum_76 = async_compile.cpp('''
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


cpp_fused_77 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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


cpp_fused_view_78 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_view_80 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_81 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_gelu_gelu_backward_sum_82 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_83 = async_compile.cpp('''
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


cpp_fused_85 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_view_88 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_89 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 * tmp7;
                        auto tmp10 = tmp8 * tmp9;
                        tmp8.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr7[static_cast<long>(x0)];
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr3[static_cast<long>(x0)];
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (768L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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


cpp_fused_gelu_gelu_backward_sum_90 = async_compile.cpp('''
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


cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_91 = async_compile.cpp('''
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


cpp_fused_sum_92 = async_compile.cpp('''
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


cpp_fused_93 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_view_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
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


cpp_fused_sum_view_96 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_sum_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
    auto out_ptr3 = in_out_ptr1;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    tmp23.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    tmp28.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = static_cast<float>(0.0);
                    auto tmp1 = at::vec::Vectorized<float>(tmp0);
                    tmp1.store(out_ptr8 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr3[static_cast<long>(x0)];
                auto tmp1 = static_cast<bool>(0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 ? tmp2 : tmp0;
                in_out_ptr1[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_embedding_dense_backward_102 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_embedding_dense_backward_103 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_embedding_dense_backward_104 = async_compile.cpp('''
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
    primals_8, primals_18, primals_24, primals_34, primals_40, primals_50, primals_56, primals_66, primals_72, primals_82, primals_88, primals_98, primals_104, primals_114, primals_120, primals_130, primals_136, primals_146, primals_152, primals_162, primals_168, primals_178, primals_184, primals_194, primals_200, primals_207, full_default, slice_1, select, select_1, select_2, select_3, mul_1, getitem_3, view, getitem_149, permute_default_67, permute_default_68, alias_default_23, permute_default_69, permute_default_70, view_16, getitem_7, mul_3, view_18, addmm_4, view_20, getitem_11, mul_8, view_22, getitem_147, permute_default_61, permute_default_62, alias_default_21, permute_default_63, permute_default_64, view_38, getitem_17, mul_10, view_40, addmm_10, view_42, getitem_21, mul_15, view_44, getitem_145, permute_default_55, permute_default_56, alias_default_19, permute_default_57, permute_default_58, view_60, getitem_27, mul_17, view_62, addmm_16, view_64, getitem_31, mul_22, view_66, getitem_143, permute_default_49, permute_default_50, alias_default_17, permute_default_51, permute_default_52, view_82, getitem_37, mul_24, view_84, addmm_22, view_86, getitem_41, mul_29, view_88, getitem_141, permute_default_43, permute_default_44, alias_default_15, permute_default_45, permute_default_46, view_104, getitem_47, mul_31, view_106, addmm_28, view_108, getitem_51, mul_36, view_110, getitem_139, permute_default_37, permute_default_38, alias_default_13, permute_default_39, permute_default_40, view_126, getitem_57, mul_38, view_128, addmm_34, view_130, getitem_61, mul_43, view_132, getitem_137, permute_default_31, permute_default_32, alias_default_11, permute_default_33, permute_default_34, view_148, getitem_67, mul_45, view_150, addmm_40, view_152, getitem_71, mul_50, view_154, getitem_135, permute_default_25, permute_default_26, alias_default_9, permute_default_27, permute_default_28, view_170, getitem_77, mul_52, view_172, addmm_46, view_174, getitem_81, mul_57, view_176, getitem_133, permute_default_19, permute_default_20, alias_default_7, permute_default_21, permute_default_22, view_192, getitem_87, mul_59, view_194, addmm_52, view_196, getitem_91, mul_64, view_198, getitem_131, permute_default_13, permute_default_14, alias_default_5, permute_default_15, permute_default_16, view_214, getitem_97, mul_66, view_216, addmm_58, view_218, getitem_101, mul_71, view_220, getitem_129, permute_default_7, permute_default_8, alias_default_3, permute_default_9, permute_default_10, view_236, getitem_107, mul_73, view_238, addmm_64, view_240, getitem_111, mul_78, view_242, getitem_127, permute_default_1, permute_default_2, alias_default_1, permute_default_3, permute_default_4, view_258, getitem_117, mul_80, view_260, addmm_70, view_262, getitem_121, mul_85, select_8, tanh, getitem_124, getitem_125, permute_134, permute_138, div_24, permute_142, permute_146, div_25, permute_150, permute_162, permute_167, permute_171, div_27, permute_175, permute_179, div_28, permute_183, permute_195, permute_200, permute_204, div_30, permute_208, permute_212, div_31, permute_216, permute_228, permute_233, permute_237, div_33, permute_241, permute_245, div_34, permute_249, permute_261, permute_266, permute_270, div_36, permute_274, permute_278, div_37, permute_282, permute_294, permute_299, permute_303, div_39, permute_307, permute_311, div_40, permute_315, permute_327, permute_332, permute_336, div_42, permute_340, permute_344, div_43, permute_348, permute_360, permute_365, permute_369, div_45, permute_373, permute_377, div_46, permute_381, permute_393, permute_398, permute_402, div_48, permute_406, permute_410, div_49, permute_414, permute_426, permute_431, permute_435, div_51, permute_439, permute_443, div_52, permute_447, permute_459, permute_464, permute_468, div_54, permute_472, permute_476, div_55, permute_480, permute_492, permute_497, permute_501, div_57, permute_505, permute_509, div_58, permute_513, permute_525, permute_530, permute_534, div_60, tangents_1, tangents_2, tangents_3 = args
    args.clear()
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_50, (768, ), (1, ))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_136, (768, ), (1, ))
    assert_size_stride(primals_146, (768, ), (1, ))
    assert_size_stride(primals_152, (768, ), (1, ))
    assert_size_stride(primals_162, (768, ), (1, ))
    assert_size_stride(primals_168, (768, ), (1, ))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_184, (768, ), (1, ))
    assert_size_stride(primals_194, (768, ), (1, ))
    assert_size_stride(primals_200, (768, ), (1, ))
    assert_size_stride(primals_207, (1, 512), (512, 1))
    assert_size_stride(full_default, (1, 512), (512, 1))
    assert_size_stride(slice_1, (1, 512), (512, 1))
    assert_size_stride(select, (1, 512), (0, 4))
    assert_size_stride(select_1, (1, 512), (0, 4))
    assert_size_stride(select_2, (1, 512), (0, 4))
    assert_size_stride(select_3, (1, 512), (0, 4))
    assert_size_stride(mul_1, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(getitem_3, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view, (512, 768), (768, 1))
    assert_size_stride(getitem_149, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_67, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_68, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_23, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_69, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_70, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_16, (512, 768), (768, 1))
    assert_size_stride(getitem_7, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_3, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_18, (512, 768), (768, 1))
    assert_size_stride(addmm_4, (512, 3072), (3072, 1))
    assert_size_stride(view_20, (512, 3072), (3072, 1))
    assert_size_stride(getitem_11, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_8, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_22, (512, 768), (768, 1))
    assert_size_stride(getitem_147, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_61, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_62, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_21, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_63, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_64, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_38, (512, 768), (768, 1))
    assert_size_stride(getitem_17, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_10, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_40, (512, 768), (768, 1))
    assert_size_stride(addmm_10, (512, 3072), (3072, 1))
    assert_size_stride(view_42, (512, 3072), (3072, 1))
    assert_size_stride(getitem_21, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_15, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_44, (512, 768), (768, 1))
    assert_size_stride(getitem_145, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_55, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_56, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_19, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_57, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_58, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_60, (512, 768), (768, 1))
    assert_size_stride(getitem_27, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_17, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_62, (512, 768), (768, 1))
    assert_size_stride(addmm_16, (512, 3072), (3072, 1))
    assert_size_stride(view_64, (512, 3072), (3072, 1))
    assert_size_stride(getitem_31, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_22, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_66, (512, 768), (768, 1))
    assert_size_stride(getitem_143, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_49, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_50, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_17, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_51, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_52, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_82, (512, 768), (768, 1))
    assert_size_stride(getitem_37, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_24, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_84, (512, 768), (768, 1))
    assert_size_stride(addmm_22, (512, 3072), (3072, 1))
    assert_size_stride(view_86, (512, 3072), (3072, 1))
    assert_size_stride(getitem_41, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_29, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_88, (512, 768), (768, 1))
    assert_size_stride(getitem_141, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_43, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_44, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_15, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_45, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_46, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_104, (512, 768), (768, 1))
    assert_size_stride(getitem_47, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_31, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_106, (512, 768), (768, 1))
    assert_size_stride(addmm_28, (512, 3072), (3072, 1))
    assert_size_stride(view_108, (512, 3072), (3072, 1))
    assert_size_stride(getitem_51, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_36, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_110, (512, 768), (768, 1))
    assert_size_stride(getitem_139, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_37, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_38, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_13, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_39, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_40, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_126, (512, 768), (768, 1))
    assert_size_stride(getitem_57, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_38, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_128, (512, 768), (768, 1))
    assert_size_stride(addmm_34, (512, 3072), (3072, 1))
    assert_size_stride(view_130, (512, 3072), (3072, 1))
    assert_size_stride(getitem_61, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_43, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_132, (512, 768), (768, 1))
    assert_size_stride(getitem_137, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_31, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_32, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_11, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_33, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_34, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_148, (512, 768), (768, 1))
    assert_size_stride(getitem_67, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_45, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_150, (512, 768), (768, 1))
    assert_size_stride(addmm_40, (512, 3072), (3072, 1))
    assert_size_stride(view_152, (512, 3072), (3072, 1))
    assert_size_stride(getitem_71, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_50, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_154, (512, 768), (768, 1))
    assert_size_stride(getitem_135, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_25, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_26, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_9, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_27, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_28, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_170, (512, 768), (768, 1))
    assert_size_stride(getitem_77, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_52, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_172, (512, 768), (768, 1))
    assert_size_stride(addmm_46, (512, 3072), (3072, 1))
    assert_size_stride(view_174, (512, 3072), (3072, 1))
    assert_size_stride(getitem_81, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_57, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_176, (512, 768), (768, 1))
    assert_size_stride(getitem_133, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_19, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_20, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_7, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_21, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_22, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_192, (512, 768), (768, 1))
    assert_size_stride(getitem_87, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_59, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_194, (512, 768), (768, 1))
    assert_size_stride(addmm_52, (512, 3072), (3072, 1))
    assert_size_stride(view_196, (512, 3072), (3072, 1))
    assert_size_stride(getitem_91, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_64, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_198, (512, 768), (768, 1))
    assert_size_stride(getitem_131, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_13, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_14, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_5, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_15, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_16, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_214, (512, 768), (768, 1))
    assert_size_stride(getitem_97, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_66, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_216, (512, 768), (768, 1))
    assert_size_stride(addmm_58, (512, 3072), (3072, 1))
    assert_size_stride(view_218, (512, 3072), (3072, 1))
    assert_size_stride(getitem_101, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_71, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_220, (512, 768), (768, 1))
    assert_size_stride(getitem_129, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_7, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_8, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_3, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_9, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_10, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_236, (512, 768), (768, 1))
    assert_size_stride(getitem_107, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_73, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_238, (512, 768), (768, 1))
    assert_size_stride(addmm_64, (512, 3072), (3072, 1))
    assert_size_stride(view_240, (512, 3072), (3072, 1))
    assert_size_stride(getitem_111, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_78, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_242, (512, 768), (768, 1))
    assert_size_stride(getitem_127, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_1, (12, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_default_2, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_default_1, (1, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_default_3, (12, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_default_4, (12, 512, 64), (32768, 64, 1))
    assert_size_stride(view_258, (512, 768), (768, 1))
    assert_size_stride(getitem_117, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_80, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_260, (512, 768), (768, 1))
    assert_size_stride(addmm_70, (512, 3072), (3072, 1))
    assert_size_stride(view_262, (512, 3072), (3072, 1))
    assert_size_stride(getitem_121, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_85, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(select_8, (1, 768), (768, 1))
    assert_size_stride(tanh, (1, 768), (768, 1))
    assert_size_stride(getitem_124, (1, 768), (768, 1))
    assert_size_stride(getitem_125, (1, 768), (768, 1))
    assert_size_stride(permute_134, (2, 768), (768, 1))
    assert_size_stride(permute_138, (768, 768), (768, 1))
    assert_size_stride(div_24, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_142, (768, 3072), (3072, 1))
    assert_size_stride(permute_146, (3072, 768), (768, 1))
    assert_size_stride(div_25, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_150, (768, 768), (768, 1))
    assert_size_stride(permute_162, (768, 768), (768, 1))
    assert_size_stride(permute_167, (768, 768), (768, 1))
    assert_size_stride(permute_171, (768, 768), (768, 1))
    assert_size_stride(div_27, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_175, (768, 3072), (3072, 1))
    assert_size_stride(permute_179, (3072, 768), (768, 1))
    assert_size_stride(div_28, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_183, (768, 768), (768, 1))
    assert_size_stride(permute_195, (768, 768), (768, 1))
    assert_size_stride(permute_200, (768, 768), (768, 1))
    assert_size_stride(permute_204, (768, 768), (768, 1))
    assert_size_stride(div_30, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_208, (768, 3072), (3072, 1))
    assert_size_stride(permute_212, (3072, 768), (768, 1))
    assert_size_stride(div_31, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_216, (768, 768), (768, 1))
    assert_size_stride(permute_228, (768, 768), (768, 1))
    assert_size_stride(permute_233, (768, 768), (768, 1))
    assert_size_stride(permute_237, (768, 768), (768, 1))
    assert_size_stride(div_33, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_241, (768, 3072), (3072, 1))
    assert_size_stride(permute_245, (3072, 768), (768, 1))
    assert_size_stride(div_34, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_249, (768, 768), (768, 1))
    assert_size_stride(permute_261, (768, 768), (768, 1))
    assert_size_stride(permute_266, (768, 768), (768, 1))
    assert_size_stride(permute_270, (768, 768), (768, 1))
    assert_size_stride(div_36, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_274, (768, 3072), (3072, 1))
    assert_size_stride(permute_278, (3072, 768), (768, 1))
    assert_size_stride(div_37, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_282, (768, 768), (768, 1))
    assert_size_stride(permute_294, (768, 768), (768, 1))
    assert_size_stride(permute_299, (768, 768), (768, 1))
    assert_size_stride(permute_303, (768, 768), (768, 1))
    assert_size_stride(div_39, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_307, (768, 3072), (3072, 1))
    assert_size_stride(permute_311, (3072, 768), (768, 1))
    assert_size_stride(div_40, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_315, (768, 768), (768, 1))
    assert_size_stride(permute_327, (768, 768), (768, 1))
    assert_size_stride(permute_332, (768, 768), (768, 1))
    assert_size_stride(permute_336, (768, 768), (768, 1))
    assert_size_stride(div_42, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_340, (768, 3072), (3072, 1))
    assert_size_stride(permute_344, (3072, 768), (768, 1))
    assert_size_stride(div_43, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_348, (768, 768), (768, 1))
    assert_size_stride(permute_360, (768, 768), (768, 1))
    assert_size_stride(permute_365, (768, 768), (768, 1))
    assert_size_stride(permute_369, (768, 768), (768, 1))
    assert_size_stride(div_45, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_373, (768, 3072), (3072, 1))
    assert_size_stride(permute_377, (3072, 768), (768, 1))
    assert_size_stride(div_46, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_381, (768, 768), (768, 1))
    assert_size_stride(permute_393, (768, 768), (768, 1))
    assert_size_stride(permute_398, (768, 768), (768, 1))
    assert_size_stride(permute_402, (768, 768), (768, 1))
    assert_size_stride(div_48, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_406, (768, 3072), (3072, 1))
    assert_size_stride(permute_410, (3072, 768), (768, 1))
    assert_size_stride(div_49, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_414, (768, 768), (768, 1))
    assert_size_stride(permute_426, (768, 768), (768, 1))
    assert_size_stride(permute_431, (768, 768), (768, 1))
    assert_size_stride(permute_435, (768, 768), (768, 1))
    assert_size_stride(div_51, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_439, (768, 3072), (3072, 1))
    assert_size_stride(permute_443, (3072, 768), (768, 1))
    assert_size_stride(div_52, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_447, (768, 768), (768, 1))
    assert_size_stride(permute_459, (768, 768), (768, 1))
    assert_size_stride(permute_464, (768, 768), (768, 1))
    assert_size_stride(permute_468, (768, 768), (768, 1))
    assert_size_stride(div_54, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_472, (768, 3072), (3072, 1))
    assert_size_stride(permute_476, (3072, 768), (768, 1))
    assert_size_stride(div_55, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_480, (768, 768), (768, 1))
    assert_size_stride(permute_492, (768, 768), (768, 1))
    assert_size_stride(permute_497, (768, 768), (768, 1))
    assert_size_stride(permute_501, (768, 768), (768, 1))
    assert_size_stride(div_57, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_505, (768, 3072), (3072, 1))
    assert_size_stride(permute_509, (3072, 768), (768, 1))
    assert_size_stride(div_58, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_513, (768, 768), (768, 1))
    assert_size_stride(permute_525, (768, 768), (768, 1))
    assert_size_stride(permute_530, (768, 768), (768, 1))
    assert_size_stride(permute_534, (768, 768), (768, 1))
    assert_size_stride(div_60, (1, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(tangents_2, (1, 768), (768, 1))
    assert_size_stride(tangents_3, (1, 2), (2, 1))
    buf0 = empty((1, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_3, permute_134, out=buf0)
    del permute_134
    buf1 = empty((2, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_3, (2, 1), (1, 2), 0), getitem_124, out=buf1)
    del getitem_124
    buf2 = empty((2, ), device='cpu', dtype=torch.float32)
    buf3 = buf0; del buf0  # reuse
    buf6 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_dropout_backward_sum_tanh_backward_view_0(c_void_p(buf3.data_ptr()), c_void_p(tangents_3.data_ptr()), c_void_p(tangents_2.data_ptr()), c_void_p(getitem_125.data_ptr()), c_void_p(tanh.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf6.data_ptr()))
    del getitem_125
    del tangents_2
    del tangents_3
    del tanh
    buf4 = empty((1, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf3, permute_138, out=buf4)
    del permute_138
    buf5 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf3, (768, 1), (1, 768), 0), select_8, out=buf5)
    del select_8
    buf7 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf9 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf10 = reinterpret_tensor(buf3, (768, ), (1, ), 0); del buf3  # reuse
    buf11 = empty((768, ), device='cpu', dtype=torch.float32)
    buf12 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_select_backward_1(c_void_p(tangents_1.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(mul_85.data_ptr()), c_void_p(div_24.data_ptr()), c_void_p(getitem_121.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    del div_24
    del getitem_121
    del mul_85
    del primals_200
    del tangents_1
    buf13 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf12, (512, 768), (768, 1), 0), permute_142, out=buf13)
    del permute_142
    buf14 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf12, (768, 512), (1, 768), 0), view_262, out=buf14)
    del view_262
    buf15 = buf4; del buf4  # reuse
    buf16 = reinterpret_tensor(buf13, (1, 512, 3072), (1572864, 3072, 1), 0); del buf13  # reuse
    cpp_fused_gelu_gelu_backward_sum_2(c_void_p(buf16.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(addmm_70.data_ptr()), c_void_p(buf15.data_ptr()))
    del addmm_70
    buf17 = reinterpret_tensor(buf12, (512, 768), (768, 1), 0); del buf12  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf16, (512, 3072), (3072, 1), 0), permute_146, out=buf17)
    del permute_146
    buf18 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf16, (3072, 512), (1, 3072), 0), view_260, out=buf18)
    del view_260
    buf19 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf20 = buf8; del buf8  # reuse
    buf21 = buf7; del buf7  # reuse
    buf22 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf23 = empty((768, ), device='cpu', dtype=torch.float32)
    buf24 = empty((768, ), device='cpu', dtype=torch.float32)
    buf25 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_3(c_void_p(buf16.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(mul_80.data_ptr()), c_void_p(div_25.data_ptr()), c_void_p(getitem_117.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()))
    del div_25
    del getitem_117
    del mul_80
    del primals_194
    buf26 = reinterpret_tensor(buf9, (512, 768), (768, 1), 0); del buf9  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf25, (512, 768), (768, 1), 0), permute_150, out=buf26)
    del permute_150
    buf27 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf25, (768, 512), (1, 768), 0), view_258, out=buf27)
    del view_258
    buf28 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_4(c_void_p(buf25.data_ptr()), c_void_p(buf28.data_ptr()))
    buf29 = reinterpret_tensor(buf25, (12, 512, 64), (32768, 64, 1), 0); del buf25  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_1, reinterpret_tensor(buf26, (12, 512, 64), (64, 768, 1), 0), out=buf29)
    del permute_default_1
    buf30 = empty((12, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf26, (12, 512, 64), (64, 768, 1), 0), permute_default_2, out=buf30)
    del permute_default_2
    buf31 = empty_strided((1, 12, 512, 1), (6144, 512, 1, 6144), device='cpu', dtype=torch.float32)
    buf32 = reinterpret_tensor(buf30, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf30  # reuse
    cpp_fused_5(c_void_p(buf32.data_ptr()), c_void_p(getitem_127.data_ptr()), c_void_p(alias_default_1.data_ptr()), c_void_p(buf31.data_ptr()))
    del alias_default_1
    del getitem_127
    buf33 = reinterpret_tensor(buf26, (12, 64, 512), (32768, 512, 1), 0); del buf26  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_3, reinterpret_tensor(buf32, (12, 512, 512), (262144, 512, 1), 0), out=buf33)
    del permute_default_3
    buf34 = reinterpret_tensor(buf17, (12, 512, 64), (32768, 64, 1), 0); del buf17  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf32, (12, 512, 512), (262144, 512, 1), 0), permute_default_4, out=buf34)
    del permute_default_4
    buf35 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_6(c_void_p(buf29.data_ptr()), c_void_p(buf35.data_ptr()))
    buf36 = reinterpret_tensor(buf29, (512, 768), (768, 1), 0); del buf29  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf35, permute_162, out=buf36)
    del permute_162
    buf37 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf35, (768, 512), (1, 768), 0), view_242, out=buf37)
    buf38 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf39 = reinterpret_tensor(buf33, (512, 768), (1, 512), 0); del buf33  # reuse
    cpp_fused_sum_view_7(c_void_p(buf39.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf38.data_ptr()))
    buf40 = buf35; del buf35  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf39, permute_167, out=buf40)
    del permute_167
    buf41 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf39, (768, 512), (512, 1), 0), view_242, out=buf41)
    buf42 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf43 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_view_8(c_void_p(buf39.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()))
    buf44 = reinterpret_tensor(buf39, (512, 768), (768, 1), 0); del buf39  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf43, permute_171, out=buf44)
    del permute_171
    buf45 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf43, (768, 512), (1, 768), 0), view_242, out=buf45)
    del view_242
    buf46 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf47 = reinterpret_tensor(buf34, (1, 512, 768), (393216, 768, 1), 0); del buf34  # reuse
    buf48 = buf21; del buf21  # reuse
    buf49 = buf20; del buf20  # reuse
    buf50 = buf47; del buf47  # reuse
    buf51 = empty((768, ), device='cpu', dtype=torch.float32)
    buf52 = empty((768, ), device='cpu', dtype=torch.float32)
    buf53 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_9(c_void_p(buf50.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(mul_78.data_ptr()), c_void_p(div_27.data_ptr()), c_void_p(getitem_111.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()))
    del div_27
    del getitem_111
    del mul_78
    del primals_184
    buf54 = reinterpret_tensor(buf16, (512, 3072), (3072, 1), 0); del buf16  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf53, (512, 768), (768, 1), 0), permute_175, out=buf54)
    del permute_175
    buf55 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf53, (768, 512), (1, 768), 0), view_240, out=buf55)
    del view_240
    buf56 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf57 = reinterpret_tensor(buf54, (1, 512, 3072), (1572864, 3072, 1), 0); del buf54  # reuse
    cpp_fused_gelu_gelu_backward_sum_10(c_void_p(buf57.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(addmm_64.data_ptr()), c_void_p(buf56.data_ptr()))
    del addmm_64
    buf58 = reinterpret_tensor(buf53, (512, 768), (768, 1), 0); del buf53  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf57, (512, 3072), (3072, 1), 0), permute_179, out=buf58)
    del permute_179
    buf59 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf57, (3072, 512), (1, 3072), 0), view_238, out=buf59)
    del view_238
    buf60 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf61 = buf49; del buf49  # reuse
    buf62 = buf48; del buf48  # reuse
    buf63 = reinterpret_tensor(buf44, (1, 512, 768), (393216, 768, 1), 0); del buf44  # reuse
    buf64 = empty((768, ), device='cpu', dtype=torch.float32)
    buf65 = empty((768, ), device='cpu', dtype=torch.float32)
    buf66 = reinterpret_tensor(buf43, (1, 512, 768), (393216, 768, 1), 0); del buf43  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_11(c_void_p(buf57.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(mul_73.data_ptr()), c_void_p(div_28.data_ptr()), c_void_p(getitem_107.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()))
    del div_28
    del getitem_107
    del mul_73
    del primals_178
    buf67 = buf58; del buf58  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf66, (512, 768), (768, 1), 0), permute_183, out=buf67)
    del permute_183
    buf68 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf66, (768, 512), (1, 768), 0), view_236, out=buf68)
    del view_236
    buf69 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_12(c_void_p(buf66.data_ptr()), c_void_p(buf69.data_ptr()))
    buf70 = reinterpret_tensor(buf66, (12, 512, 64), (32768, 64, 1), 0); del buf66  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_7, reinterpret_tensor(buf67, (12, 512, 64), (64, 768, 1), 0), out=buf70)
    del permute_default_7
    buf71 = reinterpret_tensor(buf32, (12, 512, 512), (262144, 512, 1), 0); del buf32  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf67, (12, 512, 64), (64, 768, 1), 0), permute_default_8, out=buf71)
    del permute_default_8
    buf72 = buf31; del buf31  # reuse
    buf73 = reinterpret_tensor(buf71, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf71  # reuse
    cpp_fused_13(c_void_p(buf73.data_ptr()), c_void_p(getitem_129.data_ptr()), c_void_p(alias_default_3.data_ptr()), c_void_p(buf72.data_ptr()))
    del alias_default_3
    del getitem_129
    buf74 = reinterpret_tensor(buf67, (12, 64, 512), (32768, 512, 1), 0); del buf67  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_9, reinterpret_tensor(buf73, (12, 512, 512), (262144, 512, 1), 0), out=buf74)
    del permute_default_9
    buf75 = reinterpret_tensor(buf50, (12, 512, 64), (32768, 64, 1), 0); del buf50  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf73, (12, 512, 512), (262144, 512, 1), 0), permute_default_10, out=buf75)
    del permute_default_10
    buf76 = buf40; del buf40  # reuse
    cpp_fused_view_14(c_void_p(buf70.data_ptr()), c_void_p(buf76.data_ptr()))
    buf77 = reinterpret_tensor(buf70, (512, 768), (768, 1), 0); del buf70  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf76, permute_195, out=buf77)
    del permute_195
    buf78 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf76, (768, 512), (1, 768), 0), view_220, out=buf78)
    buf79 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf80 = reinterpret_tensor(buf74, (512, 768), (1, 512), 0); del buf74  # reuse
    cpp_fused_sum_view_15(c_void_p(buf80.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf79.data_ptr()))
    buf81 = buf76; del buf76  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf80, permute_200, out=buf81)
    del permute_200
    buf82 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf80, (768, 512), (512, 1), 0), view_220, out=buf82)
    buf83 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf84 = buf36; del buf36  # reuse
    cpp_fused_sum_view_16(c_void_p(buf80.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()))
    buf85 = reinterpret_tensor(buf80, (512, 768), (768, 1), 0); del buf80  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf84, permute_204, out=buf85)
    del permute_204
    buf86 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf84, (768, 512), (1, 768), 0), view_220, out=buf86)
    del view_220
    buf87 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf88 = reinterpret_tensor(buf75, (1, 512, 768), (393216, 768, 1), 0); del buf75  # reuse
    buf89 = buf62; del buf62  # reuse
    buf90 = buf61; del buf61  # reuse
    buf91 = buf88; del buf88  # reuse
    buf92 = empty((768, ), device='cpu', dtype=torch.float32)
    buf93 = empty((768, ), device='cpu', dtype=torch.float32)
    buf94 = buf22; del buf22  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_17(c_void_p(buf91.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(mul_71.data_ptr()), c_void_p(div_30.data_ptr()), c_void_p(getitem_101.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()))
    del div_30
    del getitem_101
    del mul_71
    del primals_168
    buf95 = reinterpret_tensor(buf57, (512, 3072), (3072, 1), 0); del buf57  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf94, (512, 768), (768, 1), 0), permute_208, out=buf95)
    del permute_208
    buf96 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf94, (768, 512), (1, 768), 0), view_218, out=buf96)
    del view_218
    buf97 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf98 = reinterpret_tensor(buf95, (1, 512, 3072), (1572864, 3072, 1), 0); del buf95  # reuse
    cpp_fused_gelu_gelu_backward_sum_18(c_void_p(buf98.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(addmm_58.data_ptr()), c_void_p(buf97.data_ptr()))
    del addmm_58
    buf99 = reinterpret_tensor(buf94, (512, 768), (768, 1), 0); del buf94  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf98, (512, 3072), (3072, 1), 0), permute_212, out=buf99)
    del permute_212
    buf100 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf98, (3072, 512), (1, 3072), 0), view_216, out=buf100)
    del view_216
    buf101 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf102 = buf90; del buf90  # reuse
    buf103 = buf89; del buf89  # reuse
    buf104 = reinterpret_tensor(buf85, (1, 512, 768), (393216, 768, 1), 0); del buf85  # reuse
    buf105 = empty((768, ), device='cpu', dtype=torch.float32)
    buf106 = empty((768, ), device='cpu', dtype=torch.float32)
    buf107 = reinterpret_tensor(buf84, (1, 512, 768), (393216, 768, 1), 0); del buf84  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_19(c_void_p(buf98.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(mul_66.data_ptr()), c_void_p(div_31.data_ptr()), c_void_p(getitem_97.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()))
    del div_31
    del getitem_97
    del mul_66
    del primals_162
    buf108 = buf99; del buf99  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (512, 768), (768, 1), 0), permute_216, out=buf108)
    del permute_216
    buf109 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (768, 512), (1, 768), 0), view_214, out=buf109)
    del view_214
    buf110 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_20(c_void_p(buf107.data_ptr()), c_void_p(buf110.data_ptr()))
    buf111 = reinterpret_tensor(buf107, (12, 512, 64), (32768, 64, 1), 0); del buf107  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_13, reinterpret_tensor(buf108, (12, 512, 64), (64, 768, 1), 0), out=buf111)
    del permute_default_13
    buf112 = reinterpret_tensor(buf73, (12, 512, 512), (262144, 512, 1), 0); del buf73  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf108, (12, 512, 64), (64, 768, 1), 0), permute_default_14, out=buf112)
    del permute_default_14
    buf113 = buf72; del buf72  # reuse
    buf114 = reinterpret_tensor(buf112, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf112  # reuse
    cpp_fused_21(c_void_p(buf114.data_ptr()), c_void_p(getitem_131.data_ptr()), c_void_p(alias_default_5.data_ptr()), c_void_p(buf113.data_ptr()))
    del alias_default_5
    del getitem_131
    buf115 = reinterpret_tensor(buf108, (12, 64, 512), (32768, 512, 1), 0); del buf108  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_15, reinterpret_tensor(buf114, (12, 512, 512), (262144, 512, 1), 0), out=buf115)
    del permute_default_15
    buf116 = reinterpret_tensor(buf91, (12, 512, 64), (32768, 64, 1), 0); del buf91  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf114, (12, 512, 512), (262144, 512, 1), 0), permute_default_16, out=buf116)
    del permute_default_16
    buf117 = buf81; del buf81  # reuse
    cpp_fused_view_22(c_void_p(buf111.data_ptr()), c_void_p(buf117.data_ptr()))
    buf118 = reinterpret_tensor(buf111, (512, 768), (768, 1), 0); del buf111  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf117, permute_228, out=buf118)
    del permute_228
    buf119 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf117, (768, 512), (1, 768), 0), view_198, out=buf119)
    buf120 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf121 = reinterpret_tensor(buf115, (512, 768), (1, 512), 0); del buf115  # reuse
    cpp_fused_sum_view_23(c_void_p(buf121.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf120.data_ptr()))
    buf122 = buf117; del buf117  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf121, permute_233, out=buf122)
    del permute_233
    buf123 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf121, (768, 512), (512, 1), 0), view_198, out=buf123)
    buf124 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf125 = buf77; del buf77  # reuse
    cpp_fused_sum_view_24(c_void_p(buf121.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()))
    buf126 = reinterpret_tensor(buf121, (512, 768), (768, 1), 0); del buf121  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf125, permute_237, out=buf126)
    del permute_237
    buf127 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf125, (768, 512), (1, 768), 0), view_198, out=buf127)
    del view_198
    buf128 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf129 = reinterpret_tensor(buf116, (1, 512, 768), (393216, 768, 1), 0); del buf116  # reuse
    buf130 = buf103; del buf103  # reuse
    buf131 = buf102; del buf102  # reuse
    buf132 = buf129; del buf129  # reuse
    buf133 = empty((768, ), device='cpu', dtype=torch.float32)
    buf134 = empty((768, ), device='cpu', dtype=torch.float32)
    buf135 = buf63; del buf63  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_25(c_void_p(buf132.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(mul_64.data_ptr()), c_void_p(div_33.data_ptr()), c_void_p(getitem_91.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()))
    del div_33
    del getitem_91
    del mul_64
    del primals_152
    buf136 = reinterpret_tensor(buf98, (512, 3072), (3072, 1), 0); del buf98  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf135, (512, 768), (768, 1), 0), permute_241, out=buf136)
    del permute_241
    buf137 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf135, (768, 512), (1, 768), 0), view_196, out=buf137)
    del view_196
    buf138 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf139 = reinterpret_tensor(buf136, (1, 512, 3072), (1572864, 3072, 1), 0); del buf136  # reuse
    cpp_fused_gelu_gelu_backward_sum_26(c_void_p(buf139.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(addmm_52.data_ptr()), c_void_p(buf138.data_ptr()))
    del addmm_52
    buf140 = reinterpret_tensor(buf135, (512, 768), (768, 1), 0); del buf135  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf139, (512, 3072), (3072, 1), 0), permute_245, out=buf140)
    del permute_245
    buf141 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf139, (3072, 512), (1, 3072), 0), view_194, out=buf141)
    del view_194
    buf142 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf143 = buf131; del buf131  # reuse
    buf144 = buf130; del buf130  # reuse
    buf145 = reinterpret_tensor(buf126, (1, 512, 768), (393216, 768, 1), 0); del buf126  # reuse
    buf146 = empty((768, ), device='cpu', dtype=torch.float32)
    buf147 = empty((768, ), device='cpu', dtype=torch.float32)
    buf148 = reinterpret_tensor(buf125, (1, 512, 768), (393216, 768, 1), 0); del buf125  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_27(c_void_p(buf139.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(mul_59.data_ptr()), c_void_p(div_34.data_ptr()), c_void_p(getitem_87.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()))
    del div_34
    del getitem_87
    del mul_59
    del primals_146
    buf149 = buf140; del buf140  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf148, (512, 768), (768, 1), 0), permute_249, out=buf149)
    del permute_249
    buf150 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf148, (768, 512), (1, 768), 0), view_192, out=buf150)
    del view_192
    buf151 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_28(c_void_p(buf148.data_ptr()), c_void_p(buf151.data_ptr()))
    buf152 = reinterpret_tensor(buf148, (12, 512, 64), (32768, 64, 1), 0); del buf148  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_19, reinterpret_tensor(buf149, (12, 512, 64), (64, 768, 1), 0), out=buf152)
    del permute_default_19
    buf153 = reinterpret_tensor(buf114, (12, 512, 512), (262144, 512, 1), 0); del buf114  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf149, (12, 512, 64), (64, 768, 1), 0), permute_default_20, out=buf153)
    del permute_default_20
    buf154 = buf113; del buf113  # reuse
    buf155 = reinterpret_tensor(buf153, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf153  # reuse
    cpp_fused_29(c_void_p(buf155.data_ptr()), c_void_p(getitem_133.data_ptr()), c_void_p(alias_default_7.data_ptr()), c_void_p(buf154.data_ptr()))
    del alias_default_7
    del getitem_133
    buf156 = reinterpret_tensor(buf149, (12, 64, 512), (32768, 512, 1), 0); del buf149  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_21, reinterpret_tensor(buf155, (12, 512, 512), (262144, 512, 1), 0), out=buf156)
    del permute_default_21
    buf157 = reinterpret_tensor(buf132, (12, 512, 64), (32768, 64, 1), 0); del buf132  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf155, (12, 512, 512), (262144, 512, 1), 0), permute_default_22, out=buf157)
    del permute_default_22
    buf158 = buf122; del buf122  # reuse
    cpp_fused_view_30(c_void_p(buf152.data_ptr()), c_void_p(buf158.data_ptr()))
    buf159 = reinterpret_tensor(buf152, (512, 768), (768, 1), 0); del buf152  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf158, permute_261, out=buf159)
    del permute_261
    buf160 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf158, (768, 512), (1, 768), 0), view_176, out=buf160)
    buf161 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf162 = reinterpret_tensor(buf156, (512, 768), (1, 512), 0); del buf156  # reuse
    cpp_fused_sum_view_31(c_void_p(buf162.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf161.data_ptr()))
    buf163 = buf158; del buf158  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf162, permute_266, out=buf163)
    del permute_266
    buf164 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf162, (768, 512), (512, 1), 0), view_176, out=buf164)
    buf165 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf166 = buf118; del buf118  # reuse
    cpp_fused_sum_view_32(c_void_p(buf162.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()))
    buf167 = reinterpret_tensor(buf162, (512, 768), (768, 1), 0); del buf162  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf166, permute_270, out=buf167)
    del permute_270
    buf168 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf166, (768, 512), (1, 768), 0), view_176, out=buf168)
    del view_176
    buf169 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf170 = reinterpret_tensor(buf157, (1, 512, 768), (393216, 768, 1), 0); del buf157  # reuse
    buf171 = buf144; del buf144  # reuse
    buf172 = buf143; del buf143  # reuse
    buf173 = buf170; del buf170  # reuse
    buf174 = empty((768, ), device='cpu', dtype=torch.float32)
    buf175 = empty((768, ), device='cpu', dtype=torch.float32)
    buf176 = buf104; del buf104  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_33(c_void_p(buf173.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(mul_57.data_ptr()), c_void_p(div_36.data_ptr()), c_void_p(getitem_81.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()))
    del div_36
    del getitem_81
    del mul_57
    del primals_136
    buf177 = reinterpret_tensor(buf139, (512, 3072), (3072, 1), 0); del buf139  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf176, (512, 768), (768, 1), 0), permute_274, out=buf177)
    del permute_274
    buf178 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf176, (768, 512), (1, 768), 0), view_174, out=buf178)
    del view_174
    buf179 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf180 = reinterpret_tensor(buf177, (1, 512, 3072), (1572864, 3072, 1), 0); del buf177  # reuse
    cpp_fused_gelu_gelu_backward_sum_34(c_void_p(buf180.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(addmm_46.data_ptr()), c_void_p(buf179.data_ptr()))
    del addmm_46
    buf181 = reinterpret_tensor(buf176, (512, 768), (768, 1), 0); del buf176  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf180, (512, 3072), (3072, 1), 0), permute_278, out=buf181)
    del permute_278
    buf182 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf180, (3072, 512), (1, 3072), 0), view_172, out=buf182)
    del view_172
    buf183 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf184 = buf172; del buf172  # reuse
    buf185 = buf171; del buf171  # reuse
    buf186 = reinterpret_tensor(buf167, (1, 512, 768), (393216, 768, 1), 0); del buf167  # reuse
    buf187 = empty((768, ), device='cpu', dtype=torch.float32)
    buf188 = empty((768, ), device='cpu', dtype=torch.float32)
    buf189 = reinterpret_tensor(buf166, (1, 512, 768), (393216, 768, 1), 0); del buf166  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_35(c_void_p(buf180.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(mul_52.data_ptr()), c_void_p(div_37.data_ptr()), c_void_p(getitem_77.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()))
    del div_37
    del getitem_77
    del mul_52
    del primals_130
    buf190 = buf181; del buf181  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf189, (512, 768), (768, 1), 0), permute_282, out=buf190)
    del permute_282
    buf191 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf189, (768, 512), (1, 768), 0), view_170, out=buf191)
    del view_170
    buf192 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_36(c_void_p(buf189.data_ptr()), c_void_p(buf192.data_ptr()))
    buf193 = reinterpret_tensor(buf189, (12, 512, 64), (32768, 64, 1), 0); del buf189  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_25, reinterpret_tensor(buf190, (12, 512, 64), (64, 768, 1), 0), out=buf193)
    del permute_default_25
    buf194 = reinterpret_tensor(buf155, (12, 512, 512), (262144, 512, 1), 0); del buf155  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf190, (12, 512, 64), (64, 768, 1), 0), permute_default_26, out=buf194)
    del permute_default_26
    buf195 = buf154; del buf154  # reuse
    buf196 = reinterpret_tensor(buf194, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf194  # reuse
    cpp_fused_37(c_void_p(buf196.data_ptr()), c_void_p(getitem_135.data_ptr()), c_void_p(alias_default_9.data_ptr()), c_void_p(buf195.data_ptr()))
    del alias_default_9
    del getitem_135
    buf197 = reinterpret_tensor(buf190, (12, 64, 512), (32768, 512, 1), 0); del buf190  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_27, reinterpret_tensor(buf196, (12, 512, 512), (262144, 512, 1), 0), out=buf197)
    del permute_default_27
    buf198 = reinterpret_tensor(buf173, (12, 512, 64), (32768, 64, 1), 0); del buf173  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf196, (12, 512, 512), (262144, 512, 1), 0), permute_default_28, out=buf198)
    del permute_default_28
    buf199 = buf163; del buf163  # reuse
    cpp_fused_view_38(c_void_p(buf193.data_ptr()), c_void_p(buf199.data_ptr()))
    buf200 = reinterpret_tensor(buf193, (512, 768), (768, 1), 0); del buf193  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf199, permute_294, out=buf200)
    del permute_294
    buf201 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf199, (768, 512), (1, 768), 0), view_154, out=buf201)
    buf202 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf203 = reinterpret_tensor(buf197, (512, 768), (1, 512), 0); del buf197  # reuse
    cpp_fused_sum_view_39(c_void_p(buf203.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf202.data_ptr()))
    buf204 = buf199; del buf199  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf203, permute_299, out=buf204)
    del permute_299
    buf205 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf203, (768, 512), (512, 1), 0), view_154, out=buf205)
    buf206 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf207 = buf159; del buf159  # reuse
    cpp_fused_sum_view_40(c_void_p(buf203.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()))
    buf208 = reinterpret_tensor(buf203, (512, 768), (768, 1), 0); del buf203  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf207, permute_303, out=buf208)
    del permute_303
    buf209 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf207, (768, 512), (1, 768), 0), view_154, out=buf209)
    del view_154
    buf210 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf211 = reinterpret_tensor(buf198, (1, 512, 768), (393216, 768, 1), 0); del buf198  # reuse
    buf212 = buf185; del buf185  # reuse
    buf213 = buf184; del buf184  # reuse
    buf214 = buf211; del buf211  # reuse
    buf215 = empty((768, ), device='cpu', dtype=torch.float32)
    buf216 = empty((768, ), device='cpu', dtype=torch.float32)
    buf217 = buf145; del buf145  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_41(c_void_p(buf214.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(mul_50.data_ptr()), c_void_p(div_39.data_ptr()), c_void_p(getitem_71.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()))
    del div_39
    del getitem_71
    del mul_50
    del primals_120
    buf218 = reinterpret_tensor(buf180, (512, 3072), (3072, 1), 0); del buf180  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf217, (512, 768), (768, 1), 0), permute_307, out=buf218)
    del permute_307
    buf219 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf217, (768, 512), (1, 768), 0), view_152, out=buf219)
    del view_152
    buf220 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf221 = reinterpret_tensor(buf218, (1, 512, 3072), (1572864, 3072, 1), 0); del buf218  # reuse
    cpp_fused_gelu_gelu_backward_sum_42(c_void_p(buf221.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(addmm_40.data_ptr()), c_void_p(buf220.data_ptr()))
    del addmm_40
    buf222 = reinterpret_tensor(buf217, (512, 768), (768, 1), 0); del buf217  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf221, (512, 3072), (3072, 1), 0), permute_311, out=buf222)
    del permute_311
    buf223 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf221, (3072, 512), (1, 3072), 0), view_150, out=buf223)
    del view_150
    buf224 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf225 = buf213; del buf213  # reuse
    buf226 = buf212; del buf212  # reuse
    buf227 = reinterpret_tensor(buf208, (1, 512, 768), (393216, 768, 1), 0); del buf208  # reuse
    buf228 = empty((768, ), device='cpu', dtype=torch.float32)
    buf229 = empty((768, ), device='cpu', dtype=torch.float32)
    buf230 = reinterpret_tensor(buf207, (1, 512, 768), (393216, 768, 1), 0); del buf207  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_43(c_void_p(buf221.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(mul_45.data_ptr()), c_void_p(div_40.data_ptr()), c_void_p(getitem_67.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()))
    del div_40
    del getitem_67
    del mul_45
    del primals_114
    buf231 = buf222; del buf222  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf230, (512, 768), (768, 1), 0), permute_315, out=buf231)
    del permute_315
    buf232 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf230, (768, 512), (1, 768), 0), view_148, out=buf232)
    del view_148
    buf233 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_44(c_void_p(buf230.data_ptr()), c_void_p(buf233.data_ptr()))
    buf234 = reinterpret_tensor(buf230, (12, 512, 64), (32768, 64, 1), 0); del buf230  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_31, reinterpret_tensor(buf231, (12, 512, 64), (64, 768, 1), 0), out=buf234)
    del permute_default_31
    buf235 = reinterpret_tensor(buf196, (12, 512, 512), (262144, 512, 1), 0); del buf196  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf231, (12, 512, 64), (64, 768, 1), 0), permute_default_32, out=buf235)
    del permute_default_32
    buf236 = buf195; del buf195  # reuse
    buf237 = reinterpret_tensor(buf235, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf235  # reuse
    cpp_fused_45(c_void_p(buf237.data_ptr()), c_void_p(getitem_137.data_ptr()), c_void_p(alias_default_11.data_ptr()), c_void_p(buf236.data_ptr()))
    del alias_default_11
    del getitem_137
    buf238 = reinterpret_tensor(buf231, (12, 64, 512), (32768, 512, 1), 0); del buf231  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_33, reinterpret_tensor(buf237, (12, 512, 512), (262144, 512, 1), 0), out=buf238)
    del permute_default_33
    buf239 = reinterpret_tensor(buf214, (12, 512, 64), (32768, 64, 1), 0); del buf214  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf237, (12, 512, 512), (262144, 512, 1), 0), permute_default_34, out=buf239)
    del permute_default_34
    buf240 = buf204; del buf204  # reuse
    cpp_fused_view_46(c_void_p(buf234.data_ptr()), c_void_p(buf240.data_ptr()))
    buf241 = reinterpret_tensor(buf234, (512, 768), (768, 1), 0); del buf234  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf240, permute_327, out=buf241)
    del permute_327
    buf242 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf240, (768, 512), (1, 768), 0), view_132, out=buf242)
    buf243 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf244 = reinterpret_tensor(buf238, (512, 768), (1, 512), 0); del buf238  # reuse
    cpp_fused_sum_view_47(c_void_p(buf244.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf243.data_ptr()))
    buf245 = buf240; del buf240  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf244, permute_332, out=buf245)
    del permute_332
    buf246 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf244, (768, 512), (512, 1), 0), view_132, out=buf246)
    buf247 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf248 = buf200; del buf200  # reuse
    cpp_fused_sum_view_48(c_void_p(buf244.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()))
    buf249 = reinterpret_tensor(buf244, (512, 768), (768, 1), 0); del buf244  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf248, permute_336, out=buf249)
    del permute_336
    buf250 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf248, (768, 512), (1, 768), 0), view_132, out=buf250)
    del view_132
    buf251 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf252 = reinterpret_tensor(buf239, (1, 512, 768), (393216, 768, 1), 0); del buf239  # reuse
    buf253 = buf226; del buf226  # reuse
    buf254 = buf225; del buf225  # reuse
    buf255 = buf252; del buf252  # reuse
    buf256 = empty((768, ), device='cpu', dtype=torch.float32)
    buf257 = empty((768, ), device='cpu', dtype=torch.float32)
    buf258 = buf186; del buf186  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_49(c_void_p(buf255.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(mul_43.data_ptr()), c_void_p(div_42.data_ptr()), c_void_p(getitem_61.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()))
    del div_42
    del getitem_61
    del mul_43
    del primals_104
    buf259 = reinterpret_tensor(buf221, (512, 3072), (3072, 1), 0); del buf221  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf258, (512, 768), (768, 1), 0), permute_340, out=buf259)
    del permute_340
    buf260 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf258, (768, 512), (1, 768), 0), view_130, out=buf260)
    del view_130
    buf261 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf262 = reinterpret_tensor(buf259, (1, 512, 3072), (1572864, 3072, 1), 0); del buf259  # reuse
    cpp_fused_gelu_gelu_backward_sum_50(c_void_p(buf262.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(buf261.data_ptr()))
    del addmm_34
    buf263 = reinterpret_tensor(buf258, (512, 768), (768, 1), 0); del buf258  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf262, (512, 3072), (3072, 1), 0), permute_344, out=buf263)
    del permute_344
    buf264 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf262, (3072, 512), (1, 3072), 0), view_128, out=buf264)
    del view_128
    buf265 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf266 = buf254; del buf254  # reuse
    buf267 = buf253; del buf253  # reuse
    buf268 = reinterpret_tensor(buf249, (1, 512, 768), (393216, 768, 1), 0); del buf249  # reuse
    buf269 = empty((768, ), device='cpu', dtype=torch.float32)
    buf270 = empty((768, ), device='cpu', dtype=torch.float32)
    buf271 = reinterpret_tensor(buf248, (1, 512, 768), (393216, 768, 1), 0); del buf248  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_51(c_void_p(buf262.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(mul_38.data_ptr()), c_void_p(div_43.data_ptr()), c_void_p(getitem_57.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()))
    del div_43
    del getitem_57
    del mul_38
    del primals_98
    buf272 = buf263; del buf263  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf271, (512, 768), (768, 1), 0), permute_348, out=buf272)
    del permute_348
    buf273 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf271, (768, 512), (1, 768), 0), view_126, out=buf273)
    del view_126
    buf274 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_52(c_void_p(buf271.data_ptr()), c_void_p(buf274.data_ptr()))
    buf275 = reinterpret_tensor(buf271, (12, 512, 64), (32768, 64, 1), 0); del buf271  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_37, reinterpret_tensor(buf272, (12, 512, 64), (64, 768, 1), 0), out=buf275)
    del permute_default_37
    buf276 = reinterpret_tensor(buf237, (12, 512, 512), (262144, 512, 1), 0); del buf237  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf272, (12, 512, 64), (64, 768, 1), 0), permute_default_38, out=buf276)
    del permute_default_38
    buf277 = buf236; del buf236  # reuse
    buf278 = reinterpret_tensor(buf276, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf276  # reuse
    cpp_fused_53(c_void_p(buf278.data_ptr()), c_void_p(getitem_139.data_ptr()), c_void_p(alias_default_13.data_ptr()), c_void_p(buf277.data_ptr()))
    del alias_default_13
    del getitem_139
    buf279 = reinterpret_tensor(buf272, (12, 64, 512), (32768, 512, 1), 0); del buf272  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_39, reinterpret_tensor(buf278, (12, 512, 512), (262144, 512, 1), 0), out=buf279)
    del permute_default_39
    buf280 = reinterpret_tensor(buf255, (12, 512, 64), (32768, 64, 1), 0); del buf255  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf278, (12, 512, 512), (262144, 512, 1), 0), permute_default_40, out=buf280)
    del permute_default_40
    buf281 = buf245; del buf245  # reuse
    cpp_fused_view_54(c_void_p(buf275.data_ptr()), c_void_p(buf281.data_ptr()))
    buf282 = reinterpret_tensor(buf275, (512, 768), (768, 1), 0); del buf275  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf281, permute_360, out=buf282)
    del permute_360
    buf283 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf281, (768, 512), (1, 768), 0), view_110, out=buf283)
    buf284 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf285 = reinterpret_tensor(buf279, (512, 768), (1, 512), 0); del buf279  # reuse
    cpp_fused_sum_view_55(c_void_p(buf285.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf284.data_ptr()))
    buf286 = buf281; del buf281  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf285, permute_365, out=buf286)
    del permute_365
    buf287 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf285, (768, 512), (512, 1), 0), view_110, out=buf287)
    buf288 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf289 = buf241; del buf241  # reuse
    cpp_fused_sum_view_56(c_void_p(buf285.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()))
    buf290 = reinterpret_tensor(buf285, (512, 768), (768, 1), 0); del buf285  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf289, permute_369, out=buf290)
    del permute_369
    buf291 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf289, (768, 512), (1, 768), 0), view_110, out=buf291)
    del view_110
    buf292 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf293 = reinterpret_tensor(buf280, (1, 512, 768), (393216, 768, 1), 0); del buf280  # reuse
    buf294 = buf267; del buf267  # reuse
    buf295 = buf266; del buf266  # reuse
    buf296 = buf293; del buf293  # reuse
    buf297 = empty((768, ), device='cpu', dtype=torch.float32)
    buf298 = empty((768, ), device='cpu', dtype=torch.float32)
    buf299 = buf227; del buf227  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_57(c_void_p(buf296.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(mul_36.data_ptr()), c_void_p(div_45.data_ptr()), c_void_p(getitem_51.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()))
    del div_45
    del getitem_51
    del mul_36
    del primals_88
    buf300 = reinterpret_tensor(buf262, (512, 3072), (3072, 1), 0); del buf262  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf299, (512, 768), (768, 1), 0), permute_373, out=buf300)
    del permute_373
    buf301 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf299, (768, 512), (1, 768), 0), view_108, out=buf301)
    del view_108
    buf302 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf303 = reinterpret_tensor(buf300, (1, 512, 3072), (1572864, 3072, 1), 0); del buf300  # reuse
    cpp_fused_gelu_gelu_backward_sum_58(c_void_p(buf303.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(addmm_28.data_ptr()), c_void_p(buf302.data_ptr()))
    del addmm_28
    buf304 = reinterpret_tensor(buf299, (512, 768), (768, 1), 0); del buf299  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf303, (512, 3072), (3072, 1), 0), permute_377, out=buf304)
    del permute_377
    buf305 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf303, (3072, 512), (1, 3072), 0), view_106, out=buf305)
    del view_106
    buf306 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf307 = buf295; del buf295  # reuse
    buf308 = buf294; del buf294  # reuse
    buf309 = reinterpret_tensor(buf290, (1, 512, 768), (393216, 768, 1), 0); del buf290  # reuse
    buf310 = empty((768, ), device='cpu', dtype=torch.float32)
    buf311 = empty((768, ), device='cpu', dtype=torch.float32)
    buf312 = reinterpret_tensor(buf289, (1, 512, 768), (393216, 768, 1), 0); del buf289  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_59(c_void_p(buf303.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(mul_31.data_ptr()), c_void_p(div_46.data_ptr()), c_void_p(getitem_47.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()))
    del div_46
    del getitem_47
    del mul_31
    del primals_82
    buf313 = buf304; del buf304  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf312, (512, 768), (768, 1), 0), permute_381, out=buf313)
    del permute_381
    buf314 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf312, (768, 512), (1, 768), 0), view_104, out=buf314)
    del view_104
    buf315 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_60(c_void_p(buf312.data_ptr()), c_void_p(buf315.data_ptr()))
    buf316 = reinterpret_tensor(buf312, (12, 512, 64), (32768, 64, 1), 0); del buf312  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_43, reinterpret_tensor(buf313, (12, 512, 64), (64, 768, 1), 0), out=buf316)
    del permute_default_43
    buf317 = reinterpret_tensor(buf278, (12, 512, 512), (262144, 512, 1), 0); del buf278  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf313, (12, 512, 64), (64, 768, 1), 0), permute_default_44, out=buf317)
    del permute_default_44
    buf318 = buf277; del buf277  # reuse
    buf319 = reinterpret_tensor(buf317, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf317  # reuse
    cpp_fused_61(c_void_p(buf319.data_ptr()), c_void_p(getitem_141.data_ptr()), c_void_p(alias_default_15.data_ptr()), c_void_p(buf318.data_ptr()))
    del alias_default_15
    del getitem_141
    buf320 = reinterpret_tensor(buf313, (12, 64, 512), (32768, 512, 1), 0); del buf313  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_45, reinterpret_tensor(buf319, (12, 512, 512), (262144, 512, 1), 0), out=buf320)
    del permute_default_45
    buf321 = reinterpret_tensor(buf296, (12, 512, 64), (32768, 64, 1), 0); del buf296  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf319, (12, 512, 512), (262144, 512, 1), 0), permute_default_46, out=buf321)
    del permute_default_46
    buf322 = buf286; del buf286  # reuse
    cpp_fused_view_62(c_void_p(buf316.data_ptr()), c_void_p(buf322.data_ptr()))
    buf323 = reinterpret_tensor(buf316, (512, 768), (768, 1), 0); del buf316  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf322, permute_393, out=buf323)
    del permute_393
    buf324 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf322, (768, 512), (1, 768), 0), view_88, out=buf324)
    buf325 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf326 = reinterpret_tensor(buf320, (512, 768), (1, 512), 0); del buf320  # reuse
    cpp_fused_sum_view_63(c_void_p(buf326.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf325.data_ptr()))
    buf327 = buf322; del buf322  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf326, permute_398, out=buf327)
    del permute_398
    buf328 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf326, (768, 512), (512, 1), 0), view_88, out=buf328)
    buf329 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf330 = buf282; del buf282  # reuse
    cpp_fused_sum_view_64(c_void_p(buf326.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf330.data_ptr()))
    buf331 = reinterpret_tensor(buf326, (512, 768), (768, 1), 0); del buf326  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf330, permute_402, out=buf331)
    del permute_402
    buf332 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf330, (768, 512), (1, 768), 0), view_88, out=buf332)
    del view_88
    buf333 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf334 = reinterpret_tensor(buf321, (1, 512, 768), (393216, 768, 1), 0); del buf321  # reuse
    buf335 = buf308; del buf308  # reuse
    buf336 = buf307; del buf307  # reuse
    buf337 = buf334; del buf334  # reuse
    buf338 = empty((768, ), device='cpu', dtype=torch.float32)
    buf339 = empty((768, ), device='cpu', dtype=torch.float32)
    buf340 = buf268; del buf268  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_65(c_void_p(buf337.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(mul_29.data_ptr()), c_void_p(div_48.data_ptr()), c_void_p(getitem_41.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()))
    del div_48
    del getitem_41
    del mul_29
    del primals_72
    buf341 = reinterpret_tensor(buf303, (512, 3072), (3072, 1), 0); del buf303  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf340, (512, 768), (768, 1), 0), permute_406, out=buf341)
    del permute_406
    buf342 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf340, (768, 512), (1, 768), 0), view_86, out=buf342)
    del view_86
    buf343 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf344 = reinterpret_tensor(buf341, (1, 512, 3072), (1572864, 3072, 1), 0); del buf341  # reuse
    cpp_fused_gelu_gelu_backward_sum_66(c_void_p(buf344.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(buf343.data_ptr()))
    del addmm_22
    buf345 = reinterpret_tensor(buf340, (512, 768), (768, 1), 0); del buf340  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf344, (512, 3072), (3072, 1), 0), permute_410, out=buf345)
    del permute_410
    buf346 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf344, (3072, 512), (1, 3072), 0), view_84, out=buf346)
    del view_84
    buf347 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf348 = buf336; del buf336  # reuse
    buf349 = buf335; del buf335  # reuse
    buf350 = reinterpret_tensor(buf331, (1, 512, 768), (393216, 768, 1), 0); del buf331  # reuse
    buf351 = empty((768, ), device='cpu', dtype=torch.float32)
    buf352 = empty((768, ), device='cpu', dtype=torch.float32)
    buf353 = reinterpret_tensor(buf330, (1, 512, 768), (393216, 768, 1), 0); del buf330  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_67(c_void_p(buf344.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(mul_24.data_ptr()), c_void_p(div_49.data_ptr()), c_void_p(getitem_37.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()))
    del div_49
    del getitem_37
    del mul_24
    del primals_66
    buf354 = buf345; del buf345  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf353, (512, 768), (768, 1), 0), permute_414, out=buf354)
    del permute_414
    buf355 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf353, (768, 512), (1, 768), 0), view_82, out=buf355)
    del view_82
    buf356 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_68(c_void_p(buf353.data_ptr()), c_void_p(buf356.data_ptr()))
    buf357 = reinterpret_tensor(buf353, (12, 512, 64), (32768, 64, 1), 0); del buf353  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_49, reinterpret_tensor(buf354, (12, 512, 64), (64, 768, 1), 0), out=buf357)
    del permute_default_49
    buf358 = reinterpret_tensor(buf319, (12, 512, 512), (262144, 512, 1), 0); del buf319  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf354, (12, 512, 64), (64, 768, 1), 0), permute_default_50, out=buf358)
    del permute_default_50
    buf359 = buf318; del buf318  # reuse
    buf360 = reinterpret_tensor(buf358, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf358  # reuse
    cpp_fused_69(c_void_p(buf360.data_ptr()), c_void_p(getitem_143.data_ptr()), c_void_p(alias_default_17.data_ptr()), c_void_p(buf359.data_ptr()))
    del alias_default_17
    del getitem_143
    buf361 = reinterpret_tensor(buf354, (12, 64, 512), (32768, 512, 1), 0); del buf354  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_51, reinterpret_tensor(buf360, (12, 512, 512), (262144, 512, 1), 0), out=buf361)
    del permute_default_51
    buf362 = reinterpret_tensor(buf337, (12, 512, 64), (32768, 64, 1), 0); del buf337  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf360, (12, 512, 512), (262144, 512, 1), 0), permute_default_52, out=buf362)
    del permute_default_52
    buf363 = buf327; del buf327  # reuse
    cpp_fused_view_70(c_void_p(buf357.data_ptr()), c_void_p(buf363.data_ptr()))
    buf364 = reinterpret_tensor(buf357, (512, 768), (768, 1), 0); del buf357  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf363, permute_426, out=buf364)
    del permute_426
    buf365 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf363, (768, 512), (1, 768), 0), view_66, out=buf365)
    buf366 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf367 = reinterpret_tensor(buf361, (512, 768), (1, 512), 0); del buf361  # reuse
    cpp_fused_sum_view_71(c_void_p(buf367.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf366.data_ptr()))
    buf368 = buf363; del buf363  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf367, permute_431, out=buf368)
    del permute_431
    buf369 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf367, (768, 512), (512, 1), 0), view_66, out=buf369)
    buf370 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf371 = buf323; del buf323  # reuse
    cpp_fused_sum_view_72(c_void_p(buf367.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()))
    buf372 = reinterpret_tensor(buf367, (512, 768), (768, 1), 0); del buf367  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf371, permute_435, out=buf372)
    del permute_435
    buf373 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf371, (768, 512), (1, 768), 0), view_66, out=buf373)
    del view_66
    buf374 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf375 = reinterpret_tensor(buf362, (1, 512, 768), (393216, 768, 1), 0); del buf362  # reuse
    buf376 = buf349; del buf349  # reuse
    buf377 = buf348; del buf348  # reuse
    buf378 = buf375; del buf375  # reuse
    buf379 = empty((768, ), device='cpu', dtype=torch.float32)
    buf380 = empty((768, ), device='cpu', dtype=torch.float32)
    buf381 = buf309; del buf309  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_73(c_void_p(buf378.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(mul_22.data_ptr()), c_void_p(div_51.data_ptr()), c_void_p(getitem_31.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()))
    del div_51
    del getitem_31
    del mul_22
    del primals_56
    buf382 = reinterpret_tensor(buf344, (512, 3072), (3072, 1), 0); del buf344  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf381, (512, 768), (768, 1), 0), permute_439, out=buf382)
    del permute_439
    buf383 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf381, (768, 512), (1, 768), 0), view_64, out=buf383)
    del view_64
    buf384 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf385 = reinterpret_tensor(buf382, (1, 512, 3072), (1572864, 3072, 1), 0); del buf382  # reuse
    cpp_fused_gelu_gelu_backward_sum_74(c_void_p(buf385.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(addmm_16.data_ptr()), c_void_p(buf384.data_ptr()))
    del addmm_16
    buf386 = reinterpret_tensor(buf381, (512, 768), (768, 1), 0); del buf381  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf385, (512, 3072), (3072, 1), 0), permute_443, out=buf386)
    del permute_443
    buf387 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf385, (3072, 512), (1, 3072), 0), view_62, out=buf387)
    del view_62
    buf388 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf389 = buf377; del buf377  # reuse
    buf390 = buf376; del buf376  # reuse
    buf391 = reinterpret_tensor(buf372, (1, 512, 768), (393216, 768, 1), 0); del buf372  # reuse
    buf392 = empty((768, ), device='cpu', dtype=torch.float32)
    buf393 = empty((768, ), device='cpu', dtype=torch.float32)
    buf394 = reinterpret_tensor(buf371, (1, 512, 768), (393216, 768, 1), 0); del buf371  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_75(c_void_p(buf385.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(mul_17.data_ptr()), c_void_p(div_52.data_ptr()), c_void_p(getitem_27.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()))
    del div_52
    del getitem_27
    del mul_17
    del primals_50
    buf395 = buf386; del buf386  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf394, (512, 768), (768, 1), 0), permute_447, out=buf395)
    del permute_447
    buf396 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf394, (768, 512), (1, 768), 0), view_60, out=buf396)
    del view_60
    buf397 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_76(c_void_p(buf394.data_ptr()), c_void_p(buf397.data_ptr()))
    buf398 = reinterpret_tensor(buf394, (12, 512, 64), (32768, 64, 1), 0); del buf394  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_55, reinterpret_tensor(buf395, (12, 512, 64), (64, 768, 1), 0), out=buf398)
    del permute_default_55
    buf399 = reinterpret_tensor(buf360, (12, 512, 512), (262144, 512, 1), 0); del buf360  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf395, (12, 512, 64), (64, 768, 1), 0), permute_default_56, out=buf399)
    del permute_default_56
    buf400 = buf359; del buf359  # reuse
    buf401 = reinterpret_tensor(buf399, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf399  # reuse
    cpp_fused_77(c_void_p(buf401.data_ptr()), c_void_p(getitem_145.data_ptr()), c_void_p(alias_default_19.data_ptr()), c_void_p(buf400.data_ptr()))
    del alias_default_19
    del getitem_145
    buf402 = reinterpret_tensor(buf395, (12, 64, 512), (32768, 512, 1), 0); del buf395  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_57, reinterpret_tensor(buf401, (12, 512, 512), (262144, 512, 1), 0), out=buf402)
    del permute_default_57
    buf403 = reinterpret_tensor(buf378, (12, 512, 64), (32768, 64, 1), 0); del buf378  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf401, (12, 512, 512), (262144, 512, 1), 0), permute_default_58, out=buf403)
    del permute_default_58
    buf404 = buf368; del buf368  # reuse
    cpp_fused_view_78(c_void_p(buf398.data_ptr()), c_void_p(buf404.data_ptr()))
    buf405 = reinterpret_tensor(buf398, (512, 768), (768, 1), 0); del buf398  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf404, permute_459, out=buf405)
    del permute_459
    buf406 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf404, (768, 512), (1, 768), 0), view_44, out=buf406)
    buf407 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf408 = reinterpret_tensor(buf402, (512, 768), (1, 512), 0); del buf402  # reuse
    cpp_fused_sum_view_79(c_void_p(buf408.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf407.data_ptr()))
    buf409 = buf404; del buf404  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf408, permute_464, out=buf409)
    del permute_464
    buf410 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf408, (768, 512), (512, 1), 0), view_44, out=buf410)
    buf411 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf412 = buf364; del buf364  # reuse
    cpp_fused_sum_view_80(c_void_p(buf408.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf412.data_ptr()))
    buf413 = reinterpret_tensor(buf408, (512, 768), (768, 1), 0); del buf408  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf412, permute_468, out=buf413)
    del permute_468
    buf414 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf412, (768, 512), (1, 768), 0), view_44, out=buf414)
    del view_44
    buf415 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf416 = reinterpret_tensor(buf403, (1, 512, 768), (393216, 768, 1), 0); del buf403  # reuse
    buf417 = buf390; del buf390  # reuse
    buf418 = buf389; del buf389  # reuse
    buf419 = buf416; del buf416  # reuse
    buf420 = empty((768, ), device='cpu', dtype=torch.float32)
    buf421 = empty((768, ), device='cpu', dtype=torch.float32)
    buf422 = buf350; del buf350  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_81(c_void_p(buf419.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(mul_15.data_ptr()), c_void_p(div_54.data_ptr()), c_void_p(getitem_21.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf422.data_ptr()))
    del div_54
    del getitem_21
    del mul_15
    del primals_40
    buf423 = reinterpret_tensor(buf385, (512, 3072), (3072, 1), 0); del buf385  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf422, (512, 768), (768, 1), 0), permute_472, out=buf423)
    del permute_472
    buf424 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf422, (768, 512), (1, 768), 0), view_42, out=buf424)
    del view_42
    buf425 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf426 = reinterpret_tensor(buf423, (1, 512, 3072), (1572864, 3072, 1), 0); del buf423  # reuse
    cpp_fused_gelu_gelu_backward_sum_82(c_void_p(buf426.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(buf425.data_ptr()))
    del addmm_10
    buf427 = reinterpret_tensor(buf422, (512, 768), (768, 1), 0); del buf422  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf426, (512, 3072), (3072, 1), 0), permute_476, out=buf427)
    del permute_476
    buf428 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf426, (3072, 512), (1, 3072), 0), view_40, out=buf428)
    del view_40
    buf429 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf430 = buf418; del buf418  # reuse
    buf431 = buf417; del buf417  # reuse
    buf432 = reinterpret_tensor(buf413, (1, 512, 768), (393216, 768, 1), 0); del buf413  # reuse
    buf433 = empty((768, ), device='cpu', dtype=torch.float32)
    buf434 = empty((768, ), device='cpu', dtype=torch.float32)
    buf435 = reinterpret_tensor(buf412, (1, 512, 768), (393216, 768, 1), 0); del buf412  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_83(c_void_p(buf426.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(mul_10.data_ptr()), c_void_p(div_55.data_ptr()), c_void_p(getitem_17.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf435.data_ptr()))
    del div_55
    del getitem_17
    del mul_10
    del primals_34
    buf436 = buf427; del buf427  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf435, (512, 768), (768, 1), 0), permute_480, out=buf436)
    del permute_480
    buf437 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf435, (768, 512), (1, 768), 0), view_38, out=buf437)
    del view_38
    buf438 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_84(c_void_p(buf435.data_ptr()), c_void_p(buf438.data_ptr()))
    buf439 = reinterpret_tensor(buf435, (12, 512, 64), (32768, 64, 1), 0); del buf435  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_61, reinterpret_tensor(buf436, (12, 512, 64), (64, 768, 1), 0), out=buf439)
    del permute_default_61
    buf440 = reinterpret_tensor(buf401, (12, 512, 512), (262144, 512, 1), 0); del buf401  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf436, (12, 512, 64), (64, 768, 1), 0), permute_default_62, out=buf440)
    del permute_default_62
    buf441 = buf400; del buf400  # reuse
    buf442 = reinterpret_tensor(buf440, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf440  # reuse
    cpp_fused_85(c_void_p(buf442.data_ptr()), c_void_p(getitem_147.data_ptr()), c_void_p(alias_default_21.data_ptr()), c_void_p(buf441.data_ptr()))
    del alias_default_21
    del getitem_147
    buf443 = reinterpret_tensor(buf436, (12, 64, 512), (32768, 512, 1), 0); del buf436  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_63, reinterpret_tensor(buf442, (12, 512, 512), (262144, 512, 1), 0), out=buf443)
    del permute_default_63
    buf444 = reinterpret_tensor(buf419, (12, 512, 64), (32768, 64, 1), 0); del buf419  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf442, (12, 512, 512), (262144, 512, 1), 0), permute_default_64, out=buf444)
    del permute_default_64
    buf445 = buf409; del buf409  # reuse
    cpp_fused_view_86(c_void_p(buf439.data_ptr()), c_void_p(buf445.data_ptr()))
    buf446 = reinterpret_tensor(buf439, (512, 768), (768, 1), 0); del buf439  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf445, permute_492, out=buf446)
    del permute_492
    buf447 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf445, (768, 512), (1, 768), 0), view_22, out=buf447)
    buf448 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf449 = reinterpret_tensor(buf443, (512, 768), (1, 512), 0); del buf443  # reuse
    cpp_fused_sum_view_87(c_void_p(buf449.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf448.data_ptr()))
    buf450 = buf445; del buf445  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf449, permute_497, out=buf450)
    del permute_497
    buf451 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf449, (768, 512), (512, 1), 0), view_22, out=buf451)
    buf452 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf453 = buf405; del buf405  # reuse
    cpp_fused_sum_view_88(c_void_p(buf449.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()))
    buf454 = reinterpret_tensor(buf449, (512, 768), (768, 1), 0); del buf449  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf453, permute_501, out=buf454)
    del permute_501
    buf455 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf453, (768, 512), (1, 768), 0), view_22, out=buf455)
    del view_22
    buf456 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf457 = reinterpret_tensor(buf444, (1, 512, 768), (393216, 768, 1), 0); del buf444  # reuse
    buf458 = buf431; del buf431  # reuse
    buf459 = buf430; del buf430  # reuse
    buf460 = buf457; del buf457  # reuse
    buf461 = empty((768, ), device='cpu', dtype=torch.float32)
    buf462 = empty((768, ), device='cpu', dtype=torch.float32)
    buf463 = buf391; del buf391  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_89(c_void_p(buf460.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(mul_8.data_ptr()), c_void_p(div_57.data_ptr()), c_void_p(getitem_11.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()))
    del div_57
    del getitem_11
    del mul_8
    del primals_24
    buf464 = reinterpret_tensor(buf426, (512, 3072), (3072, 1), 0); del buf426  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf463, (512, 768), (768, 1), 0), permute_505, out=buf464)
    del permute_505
    buf465 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf463, (768, 512), (1, 768), 0), view_20, out=buf465)
    del view_20
    buf466 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf467 = reinterpret_tensor(buf464, (1, 512, 3072), (1572864, 3072, 1), 0); del buf464  # reuse
    cpp_fused_gelu_gelu_backward_sum_90(c_void_p(buf467.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(addmm_4.data_ptr()), c_void_p(buf466.data_ptr()))
    del addmm_4
    buf468 = reinterpret_tensor(buf463, (512, 768), (768, 1), 0); del buf463  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf467, (512, 3072), (3072, 1), 0), permute_509, out=buf468)
    del permute_509
    buf469 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf467, (3072, 512), (1, 3072), 0), view_18, out=buf469)
    del view_18
    buf470 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf471 = buf459; del buf459  # reuse
    buf472 = buf458; del buf458  # reuse
    buf473 = reinterpret_tensor(buf454, (1, 512, 768), (393216, 768, 1), 0); del buf454  # reuse
    buf474 = empty((768, ), device='cpu', dtype=torch.float32)
    buf475 = empty((768, ), device='cpu', dtype=torch.float32)
    buf476 = reinterpret_tensor(buf453, (1, 512, 768), (393216, 768, 1), 0); del buf453  # reuse
    cpp_fused_add_native_dropout_backward_native_layer_norm_backward_sum_91(c_void_p(buf467.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(mul_3.data_ptr()), c_void_p(div_58.data_ptr()), c_void_p(getitem_7.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()))
    del buf467
    del div_58
    del getitem_7
    del mul_3
    del primals_18
    buf477 = buf468; del buf468  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf476, (512, 768), (768, 1), 0), permute_513, out=buf477)
    del permute_513
    buf478 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf476, (768, 512), (1, 768), 0), view_16, out=buf478)
    del view_16
    buf479 = empty((1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_sum_92(c_void_p(buf476.data_ptr()), c_void_p(buf479.data_ptr()))
    buf480 = reinterpret_tensor(buf476, (12, 512, 64), (32768, 64, 1), 0); del buf476  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_67, reinterpret_tensor(buf477, (12, 512, 64), (64, 768, 1), 0), out=buf480)
    del permute_default_67
    buf481 = reinterpret_tensor(buf442, (12, 512, 512), (262144, 512, 1), 0); del buf442  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf477, (12, 512, 64), (64, 768, 1), 0), permute_default_68, out=buf481)
    del permute_default_68
    buf482 = buf441; del buf441  # reuse
    buf483 = reinterpret_tensor(buf481, (1, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf481  # reuse
    cpp_fused_93(c_void_p(buf483.data_ptr()), c_void_p(getitem_149.data_ptr()), c_void_p(alias_default_23.data_ptr()), c_void_p(buf482.data_ptr()))
    del alias_default_23
    del buf482
    del getitem_149
    buf484 = reinterpret_tensor(buf477, (12, 64, 512), (32768, 512, 1), 0); del buf477  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(permute_default_69, reinterpret_tensor(buf483, (12, 512, 512), (262144, 512, 1), 0), out=buf484)
    del permute_default_69
    buf485 = reinterpret_tensor(buf460, (12, 512, 64), (32768, 64, 1), 0); del buf460  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf483, (12, 512, 512), (262144, 512, 1), 0), permute_default_70, out=buf485)
    del buf483
    del permute_default_70
    buf486 = buf450; del buf450  # reuse
    cpp_fused_view_94(c_void_p(buf480.data_ptr()), c_void_p(buf486.data_ptr()))
    buf487 = reinterpret_tensor(buf480, (512, 768), (768, 1), 0); del buf480  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf486, permute_525, out=buf487)
    del permute_525
    buf488 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf486, (768, 512), (1, 768), 0), view, out=buf488)
    buf489 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf490 = reinterpret_tensor(buf484, (512, 768), (1, 512), 0); del buf484  # reuse
    cpp_fused_sum_view_95(c_void_p(buf490.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf489.data_ptr()))
    buf491 = buf486; del buf486  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf490, permute_530, out=buf491)
    del permute_530
    buf492 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf490, (768, 512), (512, 1), 0), view, out=buf492)
    buf493 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf494 = buf446; del buf446  # reuse
    cpp_fused_sum_view_96(c_void_p(buf490.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf494.data_ptr()))
    buf495 = reinterpret_tensor(buf490, (512, 768), (768, 1), 0); del buf490  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf494, permute_534, out=buf495)
    del permute_534
    buf496 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf494, (768, 512), (1, 768), 0), view, out=buf496)
    del view
    buf497 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf498 = buf473; del buf473  # reuse
    buf499 = buf472; del buf472  # reuse
    buf500 = buf471; del buf471  # reuse
    buf501 = reinterpret_tensor(buf485, (1, 512, 768), (393216, 768, 1), 0); del buf485  # reuse
    buf526 = buf432; del buf432  # reuse
    buf530 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf502 = empty((768, ), device='cpu', dtype=torch.float32)
    buf503 = empty((768, ), device='cpu', dtype=torch.float32)
    buf504 = empty((2, 768), device='cpu', dtype=torch.float32)
    buf505 = buf501; del buf501  # reuse
    cpp_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_sum_97(c_void_p(buf498.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(getitem_3.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(mul_1.data_ptr()), c_void_p(div_60.data_ptr()), c_void_p(slice_1.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()))
    del buf487
    del buf491
    del buf494
    del buf495
    del buf498
    del buf499
    del buf500
    del div_60
    del getitem_3
    del mul_1
    del primals_8
    aten.index_put_(buf504, [full_default], buf505, True)
    buf508 = empty((1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_98(c_void_p(buf508.data_ptr()))
    aten.index_put_(buf508, [full_default], buf505, True)
    del full_default
    buf511 = empty((1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_99(c_void_p(buf511.data_ptr()))
    aten.index_put_(buf511, [select_3], buf505, True)
    del select_3
    buf514 = empty((1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_100(c_void_p(buf514.data_ptr()))
    aten.index_put_(buf514, [select_2], buf505, True)
    del select_2
    buf517 = empty((1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_101(c_void_p(buf517.data_ptr()))
    aten.index_put_(buf517, [select_1], buf505, True)
    del select_1
    buf513 = empty((1024, 768), device='cpu', dtype=torch.float32)
    buf520 = buf513; del buf513  # reuse
    buf521 = empty((1024, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_dense_backward_102(c_void_p(buf520.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf521.data_ptr()))
    del buf511
    aten.index_put_(buf521, [select], buf505, True)
    del select
    buf516 = buf517; del buf517  # reuse
    buf524 = buf516; del buf516  # reuse
    buf525 = reinterpret_tensor(buf505, (512, 768), (768, 1), 0); del buf505  # reuse
    cpp_fused_add_embedding_dense_backward_103(c_void_p(buf524.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf525.data_ptr()))
    del buf514
    del buf521
    aten.index_put_(buf525, [slice_1], buf526, True)
    del buf526
    del slice_1
    buf529 = empty((30522, 768), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_dense_backward_104(c_void_p(buf529.data_ptr()))
    aten.index_put_(buf529, [primals_207], buf530, True)
    del buf530
    del primals_207
    return (buf529, buf525, buf524, buf520, buf508, buf508, buf504, buf502, buf503, reinterpret_tensor(buf496, (768, 768), (768, 1), 0), reinterpret_tensor(buf497, (768, ), (1, ), 0), reinterpret_tensor(buf492, (768, 768), (768, 1), 0), reinterpret_tensor(buf493, (768, ), (1, ), 0), reinterpret_tensor(buf488, (768, 768), (768, 1), 0), reinterpret_tensor(buf489, (768, ), (1, ), 0), reinterpret_tensor(buf478, (768, 768), (768, 1), 0), reinterpret_tensor(buf479, (768, ), (1, ), 0), buf474, buf475, reinterpret_tensor(buf469, (3072, 768), (768, 1), 0), reinterpret_tensor(buf470, (3072, ), (1, ), 0), reinterpret_tensor(buf465, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf466, (768, ), (1, ), 0), buf461, buf462, reinterpret_tensor(buf455, (768, 768), (768, 1), 0), reinterpret_tensor(buf456, (768, ), (1, ), 0), reinterpret_tensor(buf451, (768, 768), (768, 1), 0), reinterpret_tensor(buf452, (768, ), (1, ), 0), reinterpret_tensor(buf447, (768, 768), (768, 1), 0), reinterpret_tensor(buf448, (768, ), (1, ), 0), reinterpret_tensor(buf437, (768, 768), (768, 1), 0), reinterpret_tensor(buf438, (768, ), (1, ), 0), buf433, buf434, reinterpret_tensor(buf428, (3072, 768), (768, 1), 0), reinterpret_tensor(buf429, (3072, ), (1, ), 0), reinterpret_tensor(buf424, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf425, (768, ), (1, ), 0), buf420, buf421, reinterpret_tensor(buf414, (768, 768), (768, 1), 0), reinterpret_tensor(buf415, (768, ), (1, ), 0), reinterpret_tensor(buf410, (768, 768), (768, 1), 0), reinterpret_tensor(buf411, (768, ), (1, ), 0), reinterpret_tensor(buf406, (768, 768), (768, 1), 0), reinterpret_tensor(buf407, (768, ), (1, ), 0), reinterpret_tensor(buf396, (768, 768), (768, 1), 0), reinterpret_tensor(buf397, (768, ), (1, ), 0), buf392, buf393, reinterpret_tensor(buf387, (3072, 768), (768, 1), 0), reinterpret_tensor(buf388, (3072, ), (1, ), 0), reinterpret_tensor(buf383, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf384, (768, ), (1, ), 0), buf379, buf380, reinterpret_tensor(buf373, (768, 768), (768, 1), 0), reinterpret_tensor(buf374, (768, ), (1, ), 0), reinterpret_tensor(buf369, (768, 768), (768, 1), 0), reinterpret_tensor(buf370, (768, ), (1, ), 0), reinterpret_tensor(buf365, (768, 768), (768, 1), 0), reinterpret_tensor(buf366, (768, ), (1, ), 0), reinterpret_tensor(buf355, (768, 768), (768, 1), 0), reinterpret_tensor(buf356, (768, ), (1, ), 0), buf351, buf352, reinterpret_tensor(buf346, (3072, 768), (768, 1), 0), reinterpret_tensor(buf347, (3072, ), (1, ), 0), reinterpret_tensor(buf342, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf343, (768, ), (1, ), 0), buf338, buf339, reinterpret_tensor(buf332, (768, 768), (768, 1), 0), reinterpret_tensor(buf333, (768, ), (1, ), 0), reinterpret_tensor(buf328, (768, 768), (768, 1), 0), reinterpret_tensor(buf329, (768, ), (1, ), 0), reinterpret_tensor(buf324, (768, 768), (768, 1), 0), reinterpret_tensor(buf325, (768, ), (1, ), 0), reinterpret_tensor(buf314, (768, 768), (768, 1), 0), reinterpret_tensor(buf315, (768, ), (1, ), 0), buf310, buf311, reinterpret_tensor(buf305, (3072, 768), (768, 1), 0), reinterpret_tensor(buf306, (3072, ), (1, ), 0), reinterpret_tensor(buf301, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf302, (768, ), (1, ), 0), buf297, buf298, reinterpret_tensor(buf291, (768, 768), (768, 1), 0), reinterpret_tensor(buf292, (768, ), (1, ), 0), reinterpret_tensor(buf287, (768, 768), (768, 1), 0), reinterpret_tensor(buf288, (768, ), (1, ), 0), reinterpret_tensor(buf283, (768, 768), (768, 1), 0), reinterpret_tensor(buf284, (768, ), (1, ), 0), reinterpret_tensor(buf273, (768, 768), (768, 1), 0), reinterpret_tensor(buf274, (768, ), (1, ), 0), buf269, buf270, reinterpret_tensor(buf264, (3072, 768), (768, 1), 0), reinterpret_tensor(buf265, (3072, ), (1, ), 0), reinterpret_tensor(buf260, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf261, (768, ), (1, ), 0), buf256, buf257, reinterpret_tensor(buf250, (768, 768), (768, 1), 0), reinterpret_tensor(buf251, (768, ), (1, ), 0), reinterpret_tensor(buf246, (768, 768), (768, 1), 0), reinterpret_tensor(buf247, (768, ), (1, ), 0), reinterpret_tensor(buf242, (768, 768), (768, 1), 0), reinterpret_tensor(buf243, (768, ), (1, ), 0), reinterpret_tensor(buf232, (768, 768), (768, 1), 0), reinterpret_tensor(buf233, (768, ), (1, ), 0), buf228, buf229, reinterpret_tensor(buf223, (3072, 768), (768, 1), 0), reinterpret_tensor(buf224, (3072, ), (1, ), 0), reinterpret_tensor(buf219, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf220, (768, ), (1, ), 0), buf215, buf216, reinterpret_tensor(buf209, (768, 768), (768, 1), 0), reinterpret_tensor(buf210, (768, ), (1, ), 0), reinterpret_tensor(buf205, (768, 768), (768, 1), 0), reinterpret_tensor(buf206, (768, ), (1, ), 0), reinterpret_tensor(buf201, (768, 768), (768, 1), 0), reinterpret_tensor(buf202, (768, ), (1, ), 0), reinterpret_tensor(buf191, (768, 768), (768, 1), 0), reinterpret_tensor(buf192, (768, ), (1, ), 0), buf187, buf188, reinterpret_tensor(buf182, (3072, 768), (768, 1), 0), reinterpret_tensor(buf183, (3072, ), (1, ), 0), reinterpret_tensor(buf178, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf179, (768, ), (1, ), 0), buf174, buf175, reinterpret_tensor(buf168, (768, 768), (768, 1), 0), reinterpret_tensor(buf169, (768, ), (1, ), 0), reinterpret_tensor(buf164, (768, 768), (768, 1), 0), reinterpret_tensor(buf165, (768, ), (1, ), 0), reinterpret_tensor(buf160, (768, 768), (768, 1), 0), reinterpret_tensor(buf161, (768, ), (1, ), 0), reinterpret_tensor(buf150, (768, 768), (768, 1), 0), reinterpret_tensor(buf151, (768, ), (1, ), 0), buf146, buf147, reinterpret_tensor(buf141, (3072, 768), (768, 1), 0), reinterpret_tensor(buf142, (3072, ), (1, ), 0), reinterpret_tensor(buf137, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf138, (768, ), (1, ), 0), buf133, buf134, reinterpret_tensor(buf127, (768, 768), (768, 1), 0), reinterpret_tensor(buf128, (768, ), (1, ), 0), reinterpret_tensor(buf123, (768, 768), (768, 1), 0), reinterpret_tensor(buf124, (768, ), (1, ), 0), reinterpret_tensor(buf119, (768, 768), (768, 1), 0), reinterpret_tensor(buf120, (768, ), (1, ), 0), reinterpret_tensor(buf109, (768, 768), (768, 1), 0), reinterpret_tensor(buf110, (768, ), (1, ), 0), buf105, buf106, reinterpret_tensor(buf100, (3072, 768), (768, 1), 0), reinterpret_tensor(buf101, (3072, ), (1, ), 0), reinterpret_tensor(buf96, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf97, (768, ), (1, ), 0), buf92, buf93, reinterpret_tensor(buf86, (768, 768), (768, 1), 0), reinterpret_tensor(buf87, (768, ), (1, ), 0), reinterpret_tensor(buf82, (768, 768), (768, 1), 0), reinterpret_tensor(buf83, (768, ), (1, ), 0), reinterpret_tensor(buf78, (768, 768), (768, 1), 0), reinterpret_tensor(buf79, (768, ), (1, ), 0), reinterpret_tensor(buf68, (768, 768), (768, 1), 0), reinterpret_tensor(buf69, (768, ), (1, ), 0), buf64, buf65, reinterpret_tensor(buf59, (3072, 768), (768, 1), 0), reinterpret_tensor(buf60, (3072, ), (1, ), 0), reinterpret_tensor(buf55, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf56, (768, ), (1, ), 0), buf51, buf52, reinterpret_tensor(buf45, (768, 768), (768, 1), 0), reinterpret_tensor(buf46, (768, ), (1, ), 0), reinterpret_tensor(buf41, (768, 768), (768, 1), 0), reinterpret_tensor(buf42, (768, ), (1, ), 0), reinterpret_tensor(buf37, (768, 768), (768, 1), 0), reinterpret_tensor(buf38, (768, ), (1, ), 0), reinterpret_tensor(buf27, (768, 768), (768, 1), 0), reinterpret_tensor(buf28, (768, ), (1, ), 0), buf23, buf24, reinterpret_tensor(buf18, (3072, 768), (768, 1), 0), reinterpret_tensor(buf19, (3072, ), (1, ), 0), reinterpret_tensor(buf14, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf15, (768, ), (1, ), 0), buf10, buf11, reinterpret_tensor(buf5, (768, 768), (768, 1), 0), buf6, reinterpret_tensor(buf1, (2, 768), (768, 1), 0), buf2, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_8 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    full_default = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    slice_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    select = rand_strided((1, 512), (0, 4), device='cpu', dtype=torch.int64)
    select_1 = rand_strided((1, 512), (0, 4), device='cpu', dtype=torch.int64)
    select_2 = rand_strided((1, 512), (0, 4), device='cpu', dtype=torch.int64)
    select_3 = rand_strided((1, 512), (0, 4), device='cpu', dtype=torch.int64)
    mul_1 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    view = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_149 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_67 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_68 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_23 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_69 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_70 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_16 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_3 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_18 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_4 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_20 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_11 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_8 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_22 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_147 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_61 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_62 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_21 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_63 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_64 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_38 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_17 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_10 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_40 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_42 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_21 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_15 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_44 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_145 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_55 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_56 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_19 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_57 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_58 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_60 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_27 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_17 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_62 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_16 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_64 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_31 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_22 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_66 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_143 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_49 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_50 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_17 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_51 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_52 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_82 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_37 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_24 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_84 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_86 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_41 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_29 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_88 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_141 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_43 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_44 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_15 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_45 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_46 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_104 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_47 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_31 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_106 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_28 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_108 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_51 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_36 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_110 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_139 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_37 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_38 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_13 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_39 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_40 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_126 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_57 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_38 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_128 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_130 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_61 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_43 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_132 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_137 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_31 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_32 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_11 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_33 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_34 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_148 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_67 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_45 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_150 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_40 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_152 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_71 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_50 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_154 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_135 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_25 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_26 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_9 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_27 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_28 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_170 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_77 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_52 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_172 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_46 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_174 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_81 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_57 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_176 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_133 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_19 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_20 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_7 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_21 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_22 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_192 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_87 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_59 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_194 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_52 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_196 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_91 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_64 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_198 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_131 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_13 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_14 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_5 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_15 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_16 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_214 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_97 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_66 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_216 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_58 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_218 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_101 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_71 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_220 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_129 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_7 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_8 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_3 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_9 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_10 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_236 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_107 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_73 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_238 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_64 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_240 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_111 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_78 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_242 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_127 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.bool)
    permute_default_1 = rand_strided((12, 512, 512), (262144, 1, 512), device='cpu', dtype=torch.float32)
    permute_default_2 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    alias_default_1 = rand_strided((1, 12, 512, 512), (3145728, 262144, 512, 1), device='cpu', dtype=torch.float32)
    permute_default_3 = rand_strided((12, 64, 512), (32768, 1, 64), device='cpu', dtype=torch.float32)
    permute_default_4 = rand_strided((12, 512, 64), (32768, 64, 1), device='cpu', dtype=torch.float32)
    view_258 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_117 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_80 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    view_260 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_70 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_262 = rand_strided((512, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    getitem_121 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.bool)
    mul_85 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    select_8 = rand_strided((1, 768), (768, 1), device='cpu', dtype=torch.float32)
    tanh = rand_strided((1, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_124 = rand_strided((1, 768), (768, 1), device='cpu', dtype=torch.float32)
    getitem_125 = rand_strided((1, 768), (768, 1), device='cpu', dtype=torch.bool)
    permute_134 = rand_strided((2, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_138 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_142 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_146 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_25 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_150 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_162 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_167 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_171 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_27 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_175 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_179 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_28 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_183 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_195 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_200 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_204 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_30 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_208 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_212 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_31 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_216 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_228 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_233 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_237 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_33 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_241 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_245 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_34 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_249 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_261 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_266 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_270 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_36 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_274 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_278 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_37 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_282 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_294 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_299 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_303 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_39 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_307 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_311 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_40 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_315 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_327 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_332 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_336 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_42 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_340 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_344 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_43 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_348 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_360 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_365 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_369 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_45 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_373 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_377 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_46 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_381 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_393 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_398 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_402 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_48 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_406 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_410 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_49 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_414 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_426 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_431 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_435 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_51 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_439 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_443 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_52 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_447 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_459 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_464 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_468 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_54 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_472 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_476 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_55 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_480 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_492 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_497 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_501 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_57 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_505 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_509 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_58 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    permute_513 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_525 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_530 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_534 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_60 = rand_strided((1, 512, 1), (512, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    tangents_2 = rand_strided((1, 768), (768, 1), device='cpu', dtype=torch.float32)
    tangents_3 = rand_strided((1, 2), (2, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_8, primals_18, primals_24, primals_34, primals_40, primals_50, primals_56, primals_66, primals_72, primals_82, primals_88, primals_98, primals_104, primals_114, primals_120, primals_130, primals_136, primals_146, primals_152, primals_162, primals_168, primals_178, primals_184, primals_194, primals_200, primals_207, full_default, slice_1, select, select_1, select_2, select_3, mul_1, getitem_3, view, getitem_149, permute_default_67, permute_default_68, alias_default_23, permute_default_69, permute_default_70, view_16, getitem_7, mul_3, view_18, addmm_4, view_20, getitem_11, mul_8, view_22, getitem_147, permute_default_61, permute_default_62, alias_default_21, permute_default_63, permute_default_64, view_38, getitem_17, mul_10, view_40, addmm_10, view_42, getitem_21, mul_15, view_44, getitem_145, permute_default_55, permute_default_56, alias_default_19, permute_default_57, permute_default_58, view_60, getitem_27, mul_17, view_62, addmm_16, view_64, getitem_31, mul_22, view_66, getitem_143, permute_default_49, permute_default_50, alias_default_17, permute_default_51, permute_default_52, view_82, getitem_37, mul_24, view_84, addmm_22, view_86, getitem_41, mul_29, view_88, getitem_141, permute_default_43, permute_default_44, alias_default_15, permute_default_45, permute_default_46, view_104, getitem_47, mul_31, view_106, addmm_28, view_108, getitem_51, mul_36, view_110, getitem_139, permute_default_37, permute_default_38, alias_default_13, permute_default_39, permute_default_40, view_126, getitem_57, mul_38, view_128, addmm_34, view_130, getitem_61, mul_43, view_132, getitem_137, permute_default_31, permute_default_32, alias_default_11, permute_default_33, permute_default_34, view_148, getitem_67, mul_45, view_150, addmm_40, view_152, getitem_71, mul_50, view_154, getitem_135, permute_default_25, permute_default_26, alias_default_9, permute_default_27, permute_default_28, view_170, getitem_77, mul_52, view_172, addmm_46, view_174, getitem_81, mul_57, view_176, getitem_133, permute_default_19, permute_default_20, alias_default_7, permute_default_21, permute_default_22, view_192, getitem_87, mul_59, view_194, addmm_52, view_196, getitem_91, mul_64, view_198, getitem_131, permute_default_13, permute_default_14, alias_default_5, permute_default_15, permute_default_16, view_214, getitem_97, mul_66, view_216, addmm_58, view_218, getitem_101, mul_71, view_220, getitem_129, permute_default_7, permute_default_8, alias_default_3, permute_default_9, permute_default_10, view_236, getitem_107, mul_73, view_238, addmm_64, view_240, getitem_111, mul_78, view_242, getitem_127, permute_default_1, permute_default_2, alias_default_1, permute_default_3, permute_default_4, view_258, getitem_117, mul_80, view_260, addmm_70, view_262, getitem_121, mul_85, select_8, tanh, getitem_124, getitem_125, permute_134, permute_138, div_24, permute_142, permute_146, div_25, permute_150, permute_162, permute_167, permute_171, div_27, permute_175, permute_179, div_28, permute_183, permute_195, permute_200, permute_204, div_30, permute_208, permute_212, div_31, permute_216, permute_228, permute_233, permute_237, div_33, permute_241, permute_245, div_34, permute_249, permute_261, permute_266, permute_270, div_36, permute_274, permute_278, div_37, permute_282, permute_294, permute_299, permute_303, div_39, permute_307, permute_311, div_40, permute_315, permute_327, permute_332, permute_336, div_42, permute_340, permute_344, div_43, permute_348, permute_360, permute_365, permute_369, div_45, permute_373, permute_377, div_46, permute_381, permute_393, permute_398, permute_402, div_48, permute_406, permute_410, div_49, permute_414, permute_426, permute_431, permute_435, div_51, permute_439, permute_443, div_52, permute_447, permute_459, permute_464, permute_468, div_54, permute_472, permute_476, div_55, permute_480, permute_492, permute_497, permute_501, div_57, permute_505, permute_509, div_58, permute_513, permute_525, permute_530, permute_534, div_60, tangents_1, tangents_2, tangents_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('LayoutLMForSequenceClassification', benchmark_compiled_module)
