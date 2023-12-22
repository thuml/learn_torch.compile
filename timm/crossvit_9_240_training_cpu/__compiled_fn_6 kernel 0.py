
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


cpp_fused_div_select_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8000L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(2.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            tmp3.store(out_ptr0 + static_cast<long>(x0));
            tmp3.store(out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_sum_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_select_backward_sum_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10)
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp3 = in_ptr1[static_cast<long>(x2 + (256L*x0))];
                            auto tmp6 = in_ptr2[static_cast<long>(x2)];
                            auto tmp8 = in_ptr3[static_cast<long>(x2 + (256L*x1) + (50432L*x0))];
                            auto tmp9 = in_ptr4[static_cast<long>(x1 + (197L*x0))];
                            auto tmp11 = in_ptr5[static_cast<long>(x1 + (197L*x0))];
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                            auto tmp13 = decltype(tmp7)(tmp7 * tmp12);
                            tmp_acc0 = tmp_acc0 + tmp7;
                            tmp_acc1 = tmp_acc1 + tmp13;
                        }
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = tmp_acc0;
                        out_ptr2[static_cast<long>(x1 + (197L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr5[static_cast<long>(x1 + (197L*x0))];
                        auto tmp6 = in_ptr1[static_cast<long>(x2 + (256L*x0))];
                        auto tmp9 = in_ptr2[static_cast<long>(x2)];
                        auto tmp12 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp14 = in_ptr3[static_cast<long>(x2 + (256L*x1) + (50432L*x0))];
                        auto tmp15 = in_ptr4[static_cast<long>(x1 + (197L*x0))];
                        auto tmp18 = out_ptr2[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(256.0);
                        auto tmp2 = tmp0 / tmp1;
                        auto tmp3 = c10::convert<int>(x1);
                        auto tmp4 = static_cast<int>(0);
                        auto tmp5 = tmp3 == tmp4;
                        auto tmp7 = static_cast<float>(0.0);
                        auto tmp8 = tmp5 ? tmp6 : tmp7;
                        auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                        auto tmp11 = decltype(tmp10)(tmp10 * tmp1);
                        auto tmp13 = decltype(tmp11)(tmp11 - tmp12);
                        auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                        auto tmp17 = decltype(tmp16)(tmp16 * tmp0);
                        auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                        auto tmp20 = decltype(tmp13)(tmp13 - tmp19);
                        auto tmp21 = decltype(tmp2)(tmp2 * tmp20);
                        out_ptr3[static_cast<long>(x2 + (256L*x1) + (50432L*x0))] = tmp21;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp3 = in_ptr1[static_cast<long>(x0 + (256L*x1))];
                            auto tmp6 = in_ptr3[static_cast<long>(x0 + (256L*x2) + (50432L*x1))];
                            auto tmp7 = in_ptr4[static_cast<long>(x2 + (197L*x1))];
                            auto tmp9 = in_ptr5[static_cast<long>(x2 + (197L*x1))];
                            auto tmp0 = c10::convert<int>(x2);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                            auto tmp11 = decltype(tmp5)(tmp5 * tmp10);
                            tmp_acc0 = tmp_acc0 + tmp11;
                            tmp_acc1 = tmp_acc1 + tmp5;
                        }
                    }
                    out_ptr4[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr5[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        float tmp_acc1 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp3 = in_ptr6[static_cast<long>(x2 + (128L*x0))];
                            auto tmp6 = in_ptr7[static_cast<long>(x2)];
                            auto tmp8 = in_ptr8[static_cast<long>(x2 + (128L*x1) + (51328L*x0))];
                            auto tmp9 = in_ptr9[static_cast<long>(x1 + (401L*x0))];
                            auto tmp11 = in_ptr10[static_cast<long>(x1 + (401L*x0))];
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                            auto tmp13 = decltype(tmp7)(tmp7 * tmp12);
                            tmp_acc0 = tmp_acc0 + tmp7;
                            tmp_acc1 = tmp_acc1 + tmp13;
                        }
                        out_ptr6[static_cast<long>(x1 + (401L*x0))] = tmp_acc0;
                        out_ptr7[static_cast<long>(x1 + (401L*x0))] = tmp_acc1;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr10[static_cast<long>(x1 + (401L*x0))];
                        auto tmp6 = in_ptr6[static_cast<long>(x2 + (128L*x0))];
                        auto tmp9 = in_ptr7[static_cast<long>(x2)];
                        auto tmp12 = out_ptr6[static_cast<long>(x1 + (401L*x0))];
                        auto tmp14 = in_ptr8[static_cast<long>(x2 + (128L*x1) + (51328L*x0))];
                        auto tmp15 = in_ptr9[static_cast<long>(x1 + (401L*x0))];
                        auto tmp18 = out_ptr7[static_cast<long>(x1 + (401L*x0))];
                        auto tmp1 = static_cast<float>(128.0);
                        auto tmp2 = tmp0 / tmp1;
                        auto tmp3 = c10::convert<int>(x1);
                        auto tmp4 = static_cast<int>(0);
                        auto tmp5 = tmp3 == tmp4;
                        auto tmp7 = static_cast<float>(0.0);
                        auto tmp8 = tmp5 ? tmp6 : tmp7;
                        auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                        auto tmp11 = decltype(tmp10)(tmp10 * tmp1);
                        auto tmp13 = decltype(tmp11)(tmp11 - tmp12);
                        auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                        auto tmp17 = decltype(tmp16)(tmp16 * tmp0);
                        auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                        auto tmp20 = decltype(tmp13)(tmp13 - tmp19);
                        auto tmp21 = decltype(tmp2)(tmp2 * tmp20);
                        out_ptr8[static_cast<long>(x2 + (128L*x1) + (51328L*x0))] = tmp21;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                        {
                            auto tmp3 = in_ptr6[static_cast<long>(x0 + (128L*x1))];
                            auto tmp6 = in_ptr8[static_cast<long>(x0 + (128L*x2) + (51328L*x1))];
                            auto tmp7 = in_ptr9[static_cast<long>(x2 + (401L*x1))];
                            auto tmp9 = in_ptr10[static_cast<long>(x2 + (401L*x1))];
                            auto tmp0 = c10::convert<int>(x2);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 == tmp1;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp2 ? tmp3 : tmp4;
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                            auto tmp11 = decltype(tmp5)(tmp5 * tmp10);
                            tmp_acc0 = tmp_acc0 + tmp11;
                            tmp_acc1 = tmp_acc1 + tmp5;
                        }
                    }
                    out_ptr9[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr10[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_sum_3 = async_compile.cpp('''
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
                       float* out_ptr5)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (50432L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = static_cast<float>(0.5);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp5 * tmp5;
                    auto tmp17 = static_cast<float>(-0.5);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp19.exp();
                    auto tmp21 = static_cast<float>(0.3989422804014327);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = tmp5 * tmp23;
                    auto tmp25 = tmp15 + tmp24;
                    auto tmp26 = tmp0 * tmp25;
                    auto tmp27 = tmp26 * tmp2;
                    auto tmp28 = tmp27 * tmp1;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp27;
                    tmp_acc1_vec = tmp_acc1_vec + tmp28;
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr4[static_cast<long>(x0)];
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = out_ptr1[static_cast<long>(x0)];
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                auto tmp11 = out_ptr2[static_cast<long>(x0)];
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
                tmp16.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (128L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
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
''')


cpp_fused_sum_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (401L*x1) + (1604L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (4L*x2) + (1604L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = tmp_acc0 + tmp2;
                    }
                    out_ptr0[static_cast<long>(x1 + (4L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (401L*x1) + (1604L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (4L*x2) + (1604L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x1 + (4L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    auto tmp6 = static_cast<float>(0.1767766952966369);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    in_out_ptr0[static_cast<long>(x2 + (401L*x1) + (1604L*x0))] = tmp7;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(401L))) + (12832L*(c10::div_floor_integer((x1 + x1_inner), 32L))) + (51328L*(c10::div_floor_integer(x0, 401L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_sum_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((401L*x1) + (401L*x1_inner) + (51328L*(c10::div_floor_integer(x0, 401L))) + (static_cast<long>(x0) % static_cast<long>(401L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                            auto tmp23 = in_ptr5[static_cast<long>(x1 + (401L*x0))];
                            auto tmp26 = in_ptr6[static_cast<long>(x1 + (401L*x0))];
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = c10::convert<int>(x1);
                            auto tmp4 = static_cast<int>(1);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp5));
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = to_float_mask(tmp5);
                            auto tmp11 = at::vec::Vectorized<float>(tmp9);
                            auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                            auto tmp13 = tmp2 + tmp12;
                            auto tmp15 = tmp13 * tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp5));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp5));
                            auto tmp19 = decltype(tmp18)::blendv(tmp11, tmp18, tmp10);
                            auto tmp20 = tmp2 + tmp19;
                            auto tmp21 = tmp20 * tmp14;
                            auto tmp24 = at::vec::Vectorized<float>(tmp23);
                            auto tmp25 = tmp22 - tmp24;
                            auto tmp27 = at::vec::Vectorized<float>(tmp26);
                            auto tmp28 = tmp25 * tmp27;
                            auto tmp29 = tmp21 * tmp28;
                            tmp_acc0_vec = tmp_acc0_vec + tmp15;
                            tmp_acc1_vec = tmp_acc1_vec + tmp29;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr0[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc0);
                        tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                        out_ptr1[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc1);
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = static_cast<float>(0.5);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp5 * tmp5;
                    auto tmp17 = static_cast<float>(-0.5);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp19.exp();
                    auto tmp21 = static_cast<float>(0.3989422804014327);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = tmp5 * tmp23;
                    auto tmp25 = tmp15 + tmp24;
                    auto tmp26 = tmp0 * tmp25;
                    auto tmp27 = tmp26 * tmp2;
                    auto tmp28 = tmp27 * tmp1;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp27;
                    tmp_acc1_vec = tmp_acc1_vec + tmp28;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr3[static_cast<long>(x0)];
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp7 = out_ptr0[static_cast<long>(x0)];
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp11 = out_ptr1[static_cast<long>(x0)];
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
                tmp16.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_view_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(197L))) + (12608L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (50432L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (197L*x1) + (788L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (4L*x2) + (788L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = tmp_acc0 + tmp2;
                    }
                    out_ptr0[static_cast<long>(x1 + (4L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (197L*x1) + (788L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (4L*x2) + (788L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x1 + (4L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    auto tmp6 = static_cast<float>(0.125);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    in_out_ptr0[static_cast<long>(x2 + (197L*x1) + (788L*x0))] = tmp7;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((197L*x1) + (197L*x1_inner) + (50432L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>(x0) % static_cast<long>(197L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                            auto tmp23 = in_ptr5[static_cast<long>(x1 + (197L*x0))];
                            auto tmp26 = in_ptr6[static_cast<long>(x1 + (197L*x0))];
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = c10::convert<int>(x1);
                            auto tmp4 = static_cast<int>(1);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp5));
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = to_float_mask(tmp5);
                            auto tmp11 = at::vec::Vectorized<float>(tmp9);
                            auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                            auto tmp13 = tmp2 + tmp12;
                            auto tmp15 = tmp13 * tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp5));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp5));
                            auto tmp19 = decltype(tmp18)::blendv(tmp11, tmp18, tmp10);
                            auto tmp20 = tmp2 + tmp19;
                            auto tmp21 = tmp20 * tmp14;
                            auto tmp24 = at::vec::Vectorized<float>(tmp23);
                            auto tmp25 = tmp22 - tmp24;
                            auto tmp27 = at::vec::Vectorized<float>(tmp26);
                            auto tmp28 = tmp25 * tmp27;
                            auto tmp29 = tmp21 * tmp28;
                            tmp_acc0_vec = tmp_acc0_vec + tmp15;
                            tmp_acc1_vec = tmp_acc1_vec + tmp29;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0);
                        tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp19 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        auto tmp23 = in_ptr5[static_cast<long>(x1 + (197L*x0))];
                        auto tmp26 = in_ptr6[static_cast<long>(x1 + (197L*x0))];
                        auto tmp29 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = c10::convert<int>(x1);
                        auto tmp4 = static_cast<int>(1);
                        auto tmp5 = tmp3 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp5));
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                        auto tmp9 = static_cast<float>(0.0);
                        auto tmp10 = to_float_mask(tmp5);
                        auto tmp11 = at::vec::Vectorized<float>(tmp9);
                        auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                        auto tmp13 = tmp2 + tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp16 = static_cast<float>(256.0);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 * tmp17;
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp22 - tmp24;
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp25 * tmp27;
                        auto tmp30 = at::vec::Vectorized<float>(tmp29);
                        auto tmp31 = tmp28 * tmp30;
                        auto tmp32 = tmp21 - tmp31;
                        auto tmp33 = tmp26 / tmp16;
                        auto tmp34 = at::vec::Vectorized<float>(tmp33);
                        auto tmp35 = tmp34 * tmp32;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = masked_load(in_ptr7 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp5));
                            return tmp37;
                        }
                        ;
                        auto tmp38 = decltype(tmp36())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp36(), to_float_mask(tmp5));
                        auto tmp39 = decltype(tmp38)::blendv(tmp11, tmp38, tmp10);
                        auto tmp40 = tmp35 + tmp39;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        tmp40.store(out_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_15 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = static_cast<float>(0.5);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp5 * tmp5;
                    auto tmp17 = static_cast<float>(-0.5);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp19.exp();
                    auto tmp21 = static_cast<float>(0.3989422804014327);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = tmp5 * tmp23;
                    auto tmp25 = tmp15 + tmp24;
                    auto tmp26 = tmp0 * tmp25;
                    auto tmp27 = tmp26 * tmp2;
                    auto tmp28 = tmp27 * tmp1;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp27;
                    tmp_acc1_vec = tmp_acc1_vec + tmp28;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp19 = in_ptr7[static_cast<long>(x1 + (401L*x0))];
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        auto tmp23 = in_ptr9[static_cast<long>(x1 + (401L*x0))];
                        auto tmp26 = in_ptr10[static_cast<long>(x1 + (401L*x0))];
                        auto tmp29 = in_ptr11[static_cast<long>(x1 + (401L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = c10::convert<int>(x1);
                        auto tmp4 = static_cast<int>(1);
                        auto tmp5 = tmp3 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = masked_load(in_ptr5 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp5));
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                        auto tmp9 = static_cast<float>(0.0);
                        auto tmp10 = to_float_mask(tmp5);
                        auto tmp11 = at::vec::Vectorized<float>(tmp9);
                        auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                        auto tmp13 = tmp2 + tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp16 = static_cast<float>(128.0);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 * tmp17;
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp22 - tmp24;
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp25 * tmp27;
                        auto tmp30 = at::vec::Vectorized<float>(tmp29);
                        auto tmp31 = tmp28 * tmp30;
                        auto tmp32 = tmp21 - tmp31;
                        auto tmp33 = tmp26 / tmp16;
                        auto tmp34 = at::vec::Vectorized<float>(tmp33);
                        auto tmp35 = tmp34 * tmp32;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = masked_load(in_ptr12 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp5));
                            return tmp37;
                        }
                        ;
                        auto tmp38 = decltype(tmp36())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp36(), to_float_mask(tmp5));
                        auto tmp39 = decltype(tmp38)::blendv(tmp11, tmp38, tmp10);
                        auto tmp40 = tmp35 + tmp39;
                        auto tmp41 = tmp3 >= tmp4;
                        auto tmp42 = [&]
                        {
                            auto tmp43 = in_ptr10[static_cast<long>(x1 + (401L*x0))];
                            auto tmp44 = static_cast<float>(128.0);
                            auto tmp45 = tmp43 / tmp44;
                            auto tmp46 = at::vec::Vectorized<float>(tmp45);
                            auto tmp47 = tmp46 * tmp32;
                            auto tmp48 = c10::convert<int>(x1);
                            auto tmp49 = static_cast<int>(1);
                            auto tmp50 = tmp48 < tmp49;
                            auto tmp52 = tmp50 & tmp41;
                            auto tmp51 = [&]
                            {
                                auto tmp53 = masked_load(in_ptr12 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp52));
                                return tmp53;
                            }
                            ;
                            auto tmp54 = decltype(tmp51())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp51(), to_float_mask(tmp52));
                            auto tmp55 = static_cast<float>(0.0);
                            auto tmp56 = to_float_mask(tmp50);
                            auto tmp57 = at::vec::Vectorized<float>(tmp55);
                            auto tmp58 = decltype(tmp54)::blendv(tmp57, tmp54, tmp56);
                            auto tmp59 = tmp47 + tmp58;
                            return tmp59;
                        }
                        ;
                        auto tmp60 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp42(), to_float_mask(tmp41));
                        auto tmp61 = to_float_mask(tmp41);
                        auto tmp62 = decltype(tmp60)::blendv(tmp11, tmp60, tmp61);
                        auto tmp63 = [&]
                        {
                            auto tmp64 = masked_load(in_ptr13 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp41));
                            return tmp64;
                        }
                        ;
                        auto tmp65 = decltype(tmp63())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp63(), to_float_mask(tmp41));
                        auto tmp66 = decltype(tmp65)::blendv(tmp11, tmp65, tmp61);
                        auto tmp67 = tmp62 + tmp66;
                        auto tmp68 = [&]
                        {
                            auto tmp69 = in_ptr14[static_cast<long>(x0)];
                            auto tmp70 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp5));
                            auto tmp71 = masked_load(in_ptr1 + static_cast<long>(x2), to_float_mask(tmp5));
                            auto tmp72 = tmp70 * tmp71;
                            auto tmp73 = static_cast<float>(128.0);
                            auto tmp74 = at::vec::Vectorized<float>(tmp73);
                            auto tmp75 = tmp72 * tmp74;
                            auto tmp76 = out_ptr0[static_cast<long>(x0)];
                            auto tmp77 = at::vec::Vectorized<float>(tmp76);
                            auto tmp78 = tmp75 - tmp77;
                            auto tmp79 = masked_load(in_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp5));
                            auto tmp80 = out_ptr1[static_cast<long>(x0)];
                            auto tmp81 = at::vec::Vectorized<float>(tmp80);
                            auto tmp82 = tmp79 * tmp81;
                            auto tmp83 = tmp78 - tmp82;
                            auto tmp84 = at::vec::Vectorized<float>(tmp69);
                            auto tmp85 = tmp84 * tmp83;
                            return tmp85;
                        }
                        ;
                        auto tmp86 = decltype(tmp68())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp68(), to_float_mask(tmp5));
                        auto tmp87 = decltype(tmp86)::blendv(tmp11, tmp86, tmp10);
                        auto tmp88 = tmp67 + tmp87;
                        tmp40.store(out_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        tmp88.store(out_ptr4 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x2) + (51328L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x2) + (51328L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0 + (128L*x2) + (51328L*x1)));
                            auto tmp15 = in_ptr9[static_cast<long>(x2 + (401L*x1))];
                            auto tmp18 = in_ptr10[static_cast<long>(x2 + (401L*x1))];
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = static_cast<int>(1);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr5 + static_cast<long>(x0 + (128L*x1)), to_float_mask(tmp5));
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = to_float_mask(tmp5);
                            auto tmp11 = at::vec::Vectorized<float>(tmp9);
                            auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                            auto tmp13 = tmp2 + tmp12;
                            auto tmp16 = at::vec::Vectorized<float>(tmp15);
                            auto tmp17 = tmp14 - tmp16;
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp21 = tmp13 * tmp20;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = masked_load(in_ptr5 + static_cast<long>(x0 + (128L*x1)), to_float_mask(tmp5));
                                return tmp23;
                            }
                            ;
                            auto tmp24 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp5));
                            auto tmp25 = decltype(tmp24)::blendv(tmp11, tmp24, tmp10);
                            auto tmp26 = tmp2 + tmp25;
                            tmp_acc0_vec = tmp_acc0_vec + tmp21;
                            tmp_acc1_vec = tmp_acc1_vec + tmp26;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_backward_sum_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (51328L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_sum_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x2) + (50432L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x2) + (50432L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x2) + (50432L*x1)));
                            auto tmp15 = in_ptr4[static_cast<long>(x2 + (197L*x1))];
                            auto tmp18 = in_ptr5[static_cast<long>(x2 + (197L*x1))];
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = static_cast<int>(1);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr2 + static_cast<long>(x0 + (256L*x1)), to_float_mask(tmp5));
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = to_float_mask(tmp5);
                            auto tmp11 = at::vec::Vectorized<float>(tmp9);
                            auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                            auto tmp13 = tmp2 + tmp12;
                            auto tmp16 = at::vec::Vectorized<float>(tmp15);
                            auto tmp17 = tmp14 - tmp16;
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp21 = tmp13 * tmp20;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = masked_load(in_ptr2 + static_cast<long>(x0 + (256L*x1)), to_float_mask(tmp5));
                                return tmp23;
                            }
                            ;
                            auto tmp24 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp5));
                            auto tmp25 = decltype(tmp24)::blendv(tmp11, tmp24, tmp10);
                            auto tmp26 = tmp2 + tmp25;
                            tmp_acc0_vec = tmp_acc0_vec + tmp21;
                            tmp_acc1_vec = tmp_acc1_vec + tmp26;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_sum_21 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (51328L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = static_cast<float>(0.5);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp5 * tmp5;
                    auto tmp17 = static_cast<float>(-0.5);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp19.exp();
                    auto tmp21 = static_cast<float>(0.3989422804014327);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = tmp5 * tmp23;
                    auto tmp25 = tmp15 + tmp24;
                    auto tmp26 = tmp0 * tmp25;
                    auto tmp27 = tmp26 * tmp2;
                    auto tmp28 = tmp27 * tmp1;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp27;
                    tmp_acc1_vec = tmp_acc1_vec + tmp28;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_out_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = to_float_mask(tmp2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp6);
                        auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp7);
                        auto tmp10 = [&]
                        {
                            auto tmp11 = in_ptr4[static_cast<long>(x1 + (197L*x0))];
                            auto tmp12 = static_cast<float>(256.0);
                            auto tmp13 = tmp11 / tmp12;
                            auto tmp14 = masked_load(in_ptr5 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp2));
                            auto tmp15 = at::vec::Vectorized<float>(tmp13);
                            auto tmp16 = tmp15 * tmp14;
                            auto tmp17 = c10::convert<int>(x1);
                            auto tmp18 = static_cast<int>(1);
                            auto tmp19 = tmp17 < tmp18;
                            auto tmp21 = tmp19 & tmp2;
                            auto tmp20 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr6 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp21));
                                return tmp22;
                            }
                            ;
                            auto tmp23 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp21));
                            auto tmp24 = static_cast<float>(0.0);
                            auto tmp25 = to_float_mask(tmp19);
                            auto tmp26 = at::vec::Vectorized<float>(tmp24);
                            auto tmp27 = decltype(tmp23)::blendv(tmp26, tmp23, tmp25);
                            auto tmp28 = tmp16 + tmp27;
                            return tmp28;
                        }
                        ;
                        auto tmp29 = decltype(tmp10())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp10(), to_float_mask(tmp2));
                        auto tmp30 = decltype(tmp29)::blendv(tmp8, tmp29, tmp7);
                        auto tmp31 = tmp9 + tmp30;
                        auto tmp32 = tmp0 < tmp1;
                        auto tmp33 = [&]
                        {
                            auto tmp34 = in_ptr7[static_cast<long>(x0)];
                            auto tmp35 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp32));
                            auto tmp36 = masked_load(in_ptr2 + static_cast<long>(x2), to_float_mask(tmp32));
                            auto tmp37 = tmp35 * tmp36;
                            auto tmp38 = static_cast<float>(256.0);
                            auto tmp39 = at::vec::Vectorized<float>(tmp38);
                            auto tmp40 = tmp37 * tmp39;
                            auto tmp41 = out_ptr1[static_cast<long>(x0)];
                            auto tmp42 = at::vec::Vectorized<float>(tmp41);
                            auto tmp43 = tmp40 - tmp42;
                            auto tmp44 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp32));
                            auto tmp45 = out_ptr2[static_cast<long>(x0)];
                            auto tmp46 = at::vec::Vectorized<float>(tmp45);
                            auto tmp47 = tmp44 * tmp46;
                            auto tmp48 = tmp43 - tmp47;
                            auto tmp49 = at::vec::Vectorized<float>(tmp34);
                            auto tmp50 = tmp49 * tmp48;
                            return tmp50;
                        }
                        ;
                        auto tmp51 = decltype(tmp33())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp33(), to_float_mask(tmp32));
                        auto tmp52 = to_float_mask(tmp32);
                        auto tmp53 = decltype(tmp51)::blendv(tmp8, tmp51, tmp52);
                        auto tmp54 = tmp31 + tmp53;
                        tmp54.store(in_out_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_backward_sum_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (50432L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_23 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_25 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-403456L) + x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-806912L) + x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (256L*x0) + (768L*x2) + (151296L*x1)));
                        }
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-403456L) + x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-806912L) + x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (256L*x0) + (768L*x2) + (151296L*x1)));
                        }
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_33 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-403456L) + x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-806912L) + x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (256L*x0) + (768L*x2) + (151296L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_37 = async_compile.cpp('''
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
                       float* out_ptr4)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp13 = in_ptr4[static_cast<long>(x0)];
                    auto tmp18 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(256.0);
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
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1231872L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_39 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(128.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (128L*x2) + (51328L*x1) + (410624L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-410624L) + x3 + (128L*x2) + (51328L*x1) + (410624L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-821248L) + x3 + (128L*x2) + (51328L*x1) + (410624L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (128L*x0) + (384L*x2) + (153984L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_42 = async_compile.cpp('''
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
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp13 = in_ptr4[static_cast<long>(x0)];
                    auto tmp18 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(128.0);
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
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_sum_43 = async_compile.cpp('''
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
                       float* out_ptr5)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (50432L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = static_cast<float>(0.5);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp5 * tmp5;
                    auto tmp17 = static_cast<float>(-0.5);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp19.exp();
                    auto tmp21 = static_cast<float>(0.3989422804014327);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = tmp5 * tmp23;
                    auto tmp25 = tmp15 + tmp24;
                    auto tmp26 = tmp0 * tmp25;
                    auto tmp27 = tmp26 * tmp2;
                    auto tmp28 = tmp27 * tmp1;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp27;
                    tmp_acc1_vec = tmp_acc1_vec + tmp28;
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr4[static_cast<long>(x0)];
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = out_ptr1[static_cast<long>(x0)];
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                auto tmp11 = out_ptr2[static_cast<long>(x0)];
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
                tmp16.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (128L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
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
''')


cpp_fused_sum_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (401L*x1) + (1604L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (4L*x2) + (1604L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = tmp_acc0 + tmp2;
                    }
                    out_ptr0[static_cast<long>(x1 + (4L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (401L*x1) + (1604L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (4L*x2) + (1604L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x1 + (4L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    auto tmp6 = static_cast<float>(0.1767766952966369);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    in_out_ptr0[static_cast<long>(x2 + (401L*x1) + (1604L*x0))] = tmp7;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(401L))) + (12832L*(c10::div_floor_integer((x1 + x1_inner), 32L))) + (51328L*(c10::div_floor_integer(x0, 401L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_sum_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((401L*x1) + (401L*x1_inner) + (51328L*(c10::div_floor_integer(x0, 401L))) + (static_cast<long>(x0) % static_cast<long>(401L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                            auto tmp23 = in_ptr5[static_cast<long>(x1 + (401L*x0))];
                            auto tmp26 = in_ptr6[static_cast<long>(x1 + (401L*x0))];
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = c10::convert<int>(x1);
                            auto tmp4 = static_cast<int>(1);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp5));
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = to_float_mask(tmp5);
                            auto tmp11 = at::vec::Vectorized<float>(tmp9);
                            auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                            auto tmp13 = tmp2 + tmp12;
                            auto tmp15 = tmp13 * tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp5));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp5));
                            auto tmp19 = decltype(tmp18)::blendv(tmp11, tmp18, tmp10);
                            auto tmp20 = tmp2 + tmp19;
                            auto tmp21 = tmp20 * tmp14;
                            auto tmp24 = at::vec::Vectorized<float>(tmp23);
                            auto tmp25 = tmp22 - tmp24;
                            auto tmp27 = at::vec::Vectorized<float>(tmp26);
                            auto tmp28 = tmp25 * tmp27;
                            auto tmp29 = tmp21 * tmp28;
                            tmp_acc0_vec = tmp_acc0_vec + tmp15;
                            tmp_acc1_vec = tmp_acc1_vec + tmp29;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr0[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc0);
                        tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                        out_ptr1[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc1);
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = static_cast<float>(0.5);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp5 * tmp5;
                    auto tmp17 = static_cast<float>(-0.5);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp19.exp();
                    auto tmp21 = static_cast<float>(0.3989422804014327);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = tmp5 * tmp23;
                    auto tmp25 = tmp15 + tmp24;
                    auto tmp26 = tmp0 * tmp25;
                    auto tmp27 = tmp26 * tmp2;
                    auto tmp28 = tmp27 * tmp1;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp27;
                    tmp_acc1_vec = tmp_acc1_vec + tmp28;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr3[static_cast<long>(x0)];
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp7 = out_ptr0[static_cast<long>(x0)];
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp11 = out_ptr1[static_cast<long>(x0)];
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
                tmp16.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_view_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(197L))) + (12608L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (50432L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (197L*x1) + (788L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (4L*x2) + (788L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = tmp_acc0 + tmp2;
                    }
                    out_ptr0[static_cast<long>(x1 + (4L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (197L*x1) + (788L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (4L*x2) + (788L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x1 + (4L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    auto tmp6 = static_cast<float>(0.125);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    in_out_ptr0[static_cast<long>(x2 + (197L*x1) + (788L*x0))] = tmp7;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((197L*x1) + (197L*x1_inner) + (50432L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>(x0) % static_cast<long>(197L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                            auto tmp23 = in_ptr5[static_cast<long>(x1 + (197L*x0))];
                            auto tmp26 = in_ptr6[static_cast<long>(x1 + (197L*x0))];
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = c10::convert<int>(x1);
                            auto tmp4 = static_cast<int>(1);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp5));
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = to_float_mask(tmp5);
                            auto tmp11 = at::vec::Vectorized<float>(tmp9);
                            auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                            auto tmp13 = tmp2 + tmp12;
                            auto tmp15 = tmp13 * tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp5));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp5));
                            auto tmp19 = decltype(tmp18)::blendv(tmp11, tmp18, tmp10);
                            auto tmp20 = tmp2 + tmp19;
                            auto tmp21 = tmp20 * tmp14;
                            auto tmp24 = at::vec::Vectorized<float>(tmp23);
                            auto tmp25 = tmp22 - tmp24;
                            auto tmp27 = at::vec::Vectorized<float>(tmp26);
                            auto tmp28 = tmp25 * tmp27;
                            auto tmp29 = tmp21 * tmp28;
                            tmp_acc0_vec = tmp_acc0_vec + tmp15;
                            tmp_acc1_vec = tmp_acc1_vec + tmp29;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0);
                        tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp19 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        auto tmp23 = in_ptr5[static_cast<long>(x1 + (197L*x0))];
                        auto tmp26 = in_ptr6[static_cast<long>(x1 + (197L*x0))];
                        auto tmp29 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = c10::convert<int>(x1);
                        auto tmp4 = static_cast<int>(1);
                        auto tmp5 = tmp3 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp5));
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                        auto tmp9 = static_cast<float>(0.0);
                        auto tmp10 = to_float_mask(tmp5);
                        auto tmp11 = at::vec::Vectorized<float>(tmp9);
                        auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                        auto tmp13 = tmp2 + tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp16 = static_cast<float>(256.0);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 * tmp17;
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp22 - tmp24;
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp25 * tmp27;
                        auto tmp30 = at::vec::Vectorized<float>(tmp29);
                        auto tmp31 = tmp28 * tmp30;
                        auto tmp32 = tmp21 - tmp31;
                        auto tmp33 = tmp26 / tmp16;
                        auto tmp34 = at::vec::Vectorized<float>(tmp33);
                        auto tmp35 = tmp34 * tmp32;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = masked_load(in_ptr7 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp5));
                            return tmp37;
                        }
                        ;
                        auto tmp38 = decltype(tmp36())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp36(), to_float_mask(tmp5));
                        auto tmp39 = decltype(tmp38)::blendv(tmp11, tmp38, tmp10);
                        auto tmp40 = tmp35 + tmp39;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        tmp40.store(out_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_55 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = static_cast<float>(0.5);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp5 * tmp5;
                    auto tmp17 = static_cast<float>(-0.5);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp19.exp();
                    auto tmp21 = static_cast<float>(0.3989422804014327);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = tmp5 * tmp23;
                    auto tmp25 = tmp15 + tmp24;
                    auto tmp26 = tmp0 * tmp25;
                    auto tmp27 = tmp26 * tmp2;
                    auto tmp28 = tmp27 * tmp1;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp27;
                    tmp_acc1_vec = tmp_acc1_vec + tmp28;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp19 = in_ptr7[static_cast<long>(x1 + (401L*x0))];
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        auto tmp23 = in_ptr9[static_cast<long>(x1 + (401L*x0))];
                        auto tmp26 = in_ptr10[static_cast<long>(x1 + (401L*x0))];
                        auto tmp29 = in_ptr11[static_cast<long>(x1 + (401L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = c10::convert<int>(x1);
                        auto tmp4 = static_cast<int>(1);
                        auto tmp5 = tmp3 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = masked_load(in_ptr5 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp5));
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                        auto tmp9 = static_cast<float>(0.0);
                        auto tmp10 = to_float_mask(tmp5);
                        auto tmp11 = at::vec::Vectorized<float>(tmp9);
                        auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                        auto tmp13 = tmp2 + tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp16 = static_cast<float>(128.0);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 * tmp17;
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp22 - tmp24;
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp25 * tmp27;
                        auto tmp30 = at::vec::Vectorized<float>(tmp29);
                        auto tmp31 = tmp28 * tmp30;
                        auto tmp32 = tmp21 - tmp31;
                        auto tmp33 = tmp26 / tmp16;
                        auto tmp34 = at::vec::Vectorized<float>(tmp33);
                        auto tmp35 = tmp34 * tmp32;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = masked_load(in_ptr12 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp5));
                            return tmp37;
                        }
                        ;
                        auto tmp38 = decltype(tmp36())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp36(), to_float_mask(tmp5));
                        auto tmp39 = decltype(tmp38)::blendv(tmp11, tmp38, tmp10);
                        auto tmp40 = tmp35 + tmp39;
                        auto tmp41 = tmp3 >= tmp4;
                        auto tmp42 = [&]
                        {
                            auto tmp43 = in_ptr10[static_cast<long>(x1 + (401L*x0))];
                            auto tmp44 = static_cast<float>(128.0);
                            auto tmp45 = tmp43 / tmp44;
                            auto tmp46 = at::vec::Vectorized<float>(tmp45);
                            auto tmp47 = tmp46 * tmp32;
                            auto tmp48 = c10::convert<int>(x1);
                            auto tmp49 = static_cast<int>(1);
                            auto tmp50 = tmp48 < tmp49;
                            auto tmp52 = tmp50 & tmp41;
                            auto tmp51 = [&]
                            {
                                auto tmp53 = masked_load(in_ptr12 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp52));
                                return tmp53;
                            }
                            ;
                            auto tmp54 = decltype(tmp51())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp51(), to_float_mask(tmp52));
                            auto tmp55 = static_cast<float>(0.0);
                            auto tmp56 = to_float_mask(tmp50);
                            auto tmp57 = at::vec::Vectorized<float>(tmp55);
                            auto tmp58 = decltype(tmp54)::blendv(tmp57, tmp54, tmp56);
                            auto tmp59 = tmp47 + tmp58;
                            return tmp59;
                        }
                        ;
                        auto tmp60 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp42(), to_float_mask(tmp41));
                        auto tmp61 = to_float_mask(tmp41);
                        auto tmp62 = decltype(tmp60)::blendv(tmp11, tmp60, tmp61);
                        auto tmp63 = [&]
                        {
                            auto tmp64 = masked_load(in_ptr13 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp41));
                            return tmp64;
                        }
                        ;
                        auto tmp65 = decltype(tmp63())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp63(), to_float_mask(tmp41));
                        auto tmp66 = decltype(tmp65)::blendv(tmp11, tmp65, tmp61);
                        auto tmp67 = tmp62 + tmp66;
                        auto tmp68 = [&]
                        {
                            auto tmp69 = in_ptr14[static_cast<long>(x0)];
                            auto tmp70 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp5));
                            auto tmp71 = masked_load(in_ptr1 + static_cast<long>(x2), to_float_mask(tmp5));
                            auto tmp72 = tmp70 * tmp71;
                            auto tmp73 = static_cast<float>(128.0);
                            auto tmp74 = at::vec::Vectorized<float>(tmp73);
                            auto tmp75 = tmp72 * tmp74;
                            auto tmp76 = out_ptr0[static_cast<long>(x0)];
                            auto tmp77 = at::vec::Vectorized<float>(tmp76);
                            auto tmp78 = tmp75 - tmp77;
                            auto tmp79 = masked_load(in_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp5));
                            auto tmp80 = out_ptr1[static_cast<long>(x0)];
                            auto tmp81 = at::vec::Vectorized<float>(tmp80);
                            auto tmp82 = tmp79 * tmp81;
                            auto tmp83 = tmp78 - tmp82;
                            auto tmp84 = at::vec::Vectorized<float>(tmp69);
                            auto tmp85 = tmp84 * tmp83;
                            return tmp85;
                        }
                        ;
                        auto tmp86 = decltype(tmp68())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp68(), to_float_mask(tmp5));
                        auto tmp87 = decltype(tmp86)::blendv(tmp11, tmp86, tmp10);
                        auto tmp88 = tmp67 + tmp87;
                        tmp40.store(out_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        tmp88.store(out_ptr4 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x2) + (51328L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x2) + (51328L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0 + (128L*x2) + (51328L*x1)));
                            auto tmp15 = in_ptr9[static_cast<long>(x2 + (401L*x1))];
                            auto tmp18 = in_ptr10[static_cast<long>(x2 + (401L*x1))];
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = static_cast<int>(1);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr5 + static_cast<long>(x0 + (128L*x1)), to_float_mask(tmp5));
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = to_float_mask(tmp5);
                            auto tmp11 = at::vec::Vectorized<float>(tmp9);
                            auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                            auto tmp13 = tmp2 + tmp12;
                            auto tmp16 = at::vec::Vectorized<float>(tmp15);
                            auto tmp17 = tmp14 - tmp16;
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp21 = tmp13 * tmp20;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = masked_load(in_ptr5 + static_cast<long>(x0 + (128L*x1)), to_float_mask(tmp5));
                                return tmp23;
                            }
                            ;
                            auto tmp24 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp5));
                            auto tmp25 = decltype(tmp24)::blendv(tmp11, tmp24, tmp10);
                            auto tmp26 = tmp2 + tmp25;
                            tmp_acc0_vec = tmp_acc0_vec + tmp21;
                            tmp_acc1_vec = tmp_acc1_vec + tmp26;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_backward_sum_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (51328L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_sum_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x2) + (50432L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x2) + (50432L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x2) + (50432L*x1)));
                            auto tmp15 = in_ptr4[static_cast<long>(x2 + (197L*x1))];
                            auto tmp18 = in_ptr5[static_cast<long>(x2 + (197L*x1))];
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = static_cast<int>(1);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr2 + static_cast<long>(x0 + (256L*x1)), to_float_mask(tmp5));
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = to_float_mask(tmp5);
                            auto tmp11 = at::vec::Vectorized<float>(tmp9);
                            auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                            auto tmp13 = tmp2 + tmp12;
                            auto tmp16 = at::vec::Vectorized<float>(tmp15);
                            auto tmp17 = tmp14 - tmp16;
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp21 = tmp13 * tmp20;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = masked_load(in_ptr2 + static_cast<long>(x0 + (256L*x1)), to_float_mask(tmp5));
                                return tmp23;
                            }
                            ;
                            auto tmp24 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp5));
                            auto tmp25 = decltype(tmp24)::blendv(tmp11, tmp24, tmp10);
                            auto tmp26 = tmp2 + tmp25;
                            tmp_acc0_vec = tmp_acc0_vec + tmp21;
                            tmp_acc1_vec = tmp_acc1_vec + tmp26;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_sum_61 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (51328L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = static_cast<float>(0.5);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp5 * tmp5;
                    auto tmp17 = static_cast<float>(-0.5);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp19.exp();
                    auto tmp21 = static_cast<float>(0.3989422804014327);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = tmp5 * tmp23;
                    auto tmp25 = tmp15 + tmp24;
                    auto tmp26 = tmp0 * tmp25;
                    auto tmp27 = tmp26 * tmp2;
                    auto tmp28 = tmp27 * tmp1;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp27;
                    tmp_acc1_vec = tmp_acc1_vec + tmp28;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_out_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = to_float_mask(tmp2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp6);
                        auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp7);
                        auto tmp10 = [&]
                        {
                            auto tmp11 = in_ptr4[static_cast<long>(x1 + (197L*x0))];
                            auto tmp12 = static_cast<float>(256.0);
                            auto tmp13 = tmp11 / tmp12;
                            auto tmp14 = masked_load(in_ptr5 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp2));
                            auto tmp15 = at::vec::Vectorized<float>(tmp13);
                            auto tmp16 = tmp15 * tmp14;
                            auto tmp17 = c10::convert<int>(x1);
                            auto tmp18 = static_cast<int>(1);
                            auto tmp19 = tmp17 < tmp18;
                            auto tmp21 = tmp19 & tmp2;
                            auto tmp20 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr6 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp21));
                                return tmp22;
                            }
                            ;
                            auto tmp23 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp21));
                            auto tmp24 = static_cast<float>(0.0);
                            auto tmp25 = to_float_mask(tmp19);
                            auto tmp26 = at::vec::Vectorized<float>(tmp24);
                            auto tmp27 = decltype(tmp23)::blendv(tmp26, tmp23, tmp25);
                            auto tmp28 = tmp16 + tmp27;
                            return tmp28;
                        }
                        ;
                        auto tmp29 = decltype(tmp10())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp10(), to_float_mask(tmp2));
                        auto tmp30 = decltype(tmp29)::blendv(tmp8, tmp29, tmp7);
                        auto tmp31 = tmp9 + tmp30;
                        auto tmp32 = tmp0 < tmp1;
                        auto tmp33 = [&]
                        {
                            auto tmp34 = in_ptr7[static_cast<long>(x0)];
                            auto tmp35 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp32));
                            auto tmp36 = masked_load(in_ptr2 + static_cast<long>(x2), to_float_mask(tmp32));
                            auto tmp37 = tmp35 * tmp36;
                            auto tmp38 = static_cast<float>(256.0);
                            auto tmp39 = at::vec::Vectorized<float>(tmp38);
                            auto tmp40 = tmp37 * tmp39;
                            auto tmp41 = out_ptr1[static_cast<long>(x0)];
                            auto tmp42 = at::vec::Vectorized<float>(tmp41);
                            auto tmp43 = tmp40 - tmp42;
                            auto tmp44 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp32));
                            auto tmp45 = out_ptr2[static_cast<long>(x0)];
                            auto tmp46 = at::vec::Vectorized<float>(tmp45);
                            auto tmp47 = tmp44 * tmp46;
                            auto tmp48 = tmp43 - tmp47;
                            auto tmp49 = at::vec::Vectorized<float>(tmp34);
                            auto tmp50 = tmp49 * tmp48;
                            return tmp50;
                        }
                        ;
                        auto tmp51 = decltype(tmp33())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp33(), to_float_mask(tmp32));
                        auto tmp52 = to_float_mask(tmp32);
                        auto tmp53 = decltype(tmp51)::blendv(tmp8, tmp51, tmp52);
                        auto tmp54 = tmp31 + tmp53;
                        tmp54.store(in_out_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_backward_sum_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (50432L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_63 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_65 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-403456L) + x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-806912L) + x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (256L*x0) + (768L*x2) + (151296L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_69 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-403456L) + x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-806912L) + x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (256L*x0) + (768L*x2) + (151296L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_72 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_73 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_75 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-403456L) + x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-806912L) + x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (256L*x0) + (768L*x2) + (151296L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_77 = async_compile.cpp('''
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
                       float* out_ptr4)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp13 = in_ptr4[static_cast<long>(x0)];
                    auto tmp18 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(256.0);
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
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1231872L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_79 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(128.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (128L*x2) + (51328L*x1) + (410624L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-410624L) + x3 + (128L*x2) + (51328L*x1) + (410624L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-821248L) + x3 + (128L*x2) + (51328L*x1) + (410624L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (128L*x0) + (384L*x2) + (153984L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_82 = async_compile.cpp('''
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
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp13 = in_ptr4[static_cast<long>(x0)];
                    auto tmp18 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = static_cast<float>(128.0);
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
                    tmp24.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_sum_83 = async_compile.cpp('''
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
                       float* out_ptr5)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (50432L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = static_cast<float>(0.5);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp5 * tmp5;
                    auto tmp17 = static_cast<float>(-0.5);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp19.exp();
                    auto tmp21 = static_cast<float>(0.3989422804014327);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = tmp5 * tmp23;
                    auto tmp25 = tmp15 + tmp24;
                    auto tmp26 = tmp0 * tmp25;
                    auto tmp27 = tmp26 * tmp2;
                    auto tmp28 = tmp27 * tmp1;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp27;
                    tmp_acc1_vec = tmp_acc1_vec + tmp28;
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr4[static_cast<long>(x0)];
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = out_ptr1[static_cast<long>(x0)];
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                auto tmp11 = out_ptr2[static_cast<long>(x0)];
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
                tmp16.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (128L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
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
''')


cpp_fused_sum_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (401L*x1) + (1604L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (4L*x2) + (1604L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = tmp_acc0 + tmp2;
                    }
                    out_ptr0[static_cast<long>(x1 + (4L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (401L*x1) + (1604L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (4L*x2) + (1604L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x1 + (4L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    auto tmp6 = static_cast<float>(0.1767766952966369);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    in_out_ptr0[static_cast<long>(x2 + (401L*x1) + (1604L*x0))] = tmp7;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(401L))) + (12832L*(c10::div_floor_integer((x1 + x1_inner), 32L))) + (51328L*(c10::div_floor_integer(x0, 401L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_sum_87 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((401L*x1) + (401L*x1_inner) + (51328L*(c10::div_floor_integer(x0, 401L))) + (static_cast<long>(x0) % static_cast<long>(401L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_88 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                            auto tmp23 = in_ptr5[static_cast<long>(x1 + (401L*x0))];
                            auto tmp26 = in_ptr6[static_cast<long>(x1 + (401L*x0))];
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = c10::convert<int>(x1);
                            auto tmp4 = static_cast<int>(1);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp5));
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = to_float_mask(tmp5);
                            auto tmp11 = at::vec::Vectorized<float>(tmp9);
                            auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                            auto tmp13 = tmp2 + tmp12;
                            auto tmp15 = tmp13 * tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr2 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp5));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp5));
                            auto tmp19 = decltype(tmp18)::blendv(tmp11, tmp18, tmp10);
                            auto tmp20 = tmp2 + tmp19;
                            auto tmp21 = tmp20 * tmp14;
                            auto tmp24 = at::vec::Vectorized<float>(tmp23);
                            auto tmp25 = tmp22 - tmp24;
                            auto tmp27 = at::vec::Vectorized<float>(tmp26);
                            auto tmp28 = tmp25 * tmp27;
                            auto tmp29 = tmp21 * tmp28;
                            tmp_acc0_vec = tmp_acc0_vec + tmp15;
                            tmp_acc1_vec = tmp_acc1_vec + tmp29;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr0[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc0);
                        tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                        out_ptr1[static_cast<long>(x1 + (401L*x0))] = static_cast<float>(tmp_acc1);
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = static_cast<float>(0.5);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp5 * tmp5;
                    auto tmp17 = static_cast<float>(-0.5);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp19.exp();
                    auto tmp21 = static_cast<float>(0.3989422804014327);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = tmp5 * tmp23;
                    auto tmp25 = tmp15 + tmp24;
                    auto tmp26 = tmp0 * tmp25;
                    auto tmp27 = tmp26 * tmp2;
                    auto tmp28 = tmp27 * tmp1;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp27;
                    tmp_acc1_vec = tmp_acc1_vec + tmp28;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr3[static_cast<long>(x0)];
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp7 = out_ptr0[static_cast<long>(x0)];
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp11 = out_ptr1[static_cast<long>(x0)];
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
                tmp16.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_view_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(197L))) + (12608L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (50432L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (197L*x1) + (788L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (4L*x2) + (788L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = tmp_acc0 + tmp2;
                    }
                    out_ptr0[static_cast<long>(x1 + (4L*x0))] = tmp_acc0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (197L*x1) + (788L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (4L*x2) + (788L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x1 + (4L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    auto tmp6 = static_cast<float>(0.125);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    in_out_ptr0[static_cast<long>(x2 + (197L*x1) + (788L*x0))] = tmp7;
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((197L*x1) + (197L*x1_inner) + (50432L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>(x0) % static_cast<long>(197L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        float tmp_acc1 = 0;
                        at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                            auto tmp23 = in_ptr5[static_cast<long>(x1 + (197L*x0))];
                            auto tmp26 = in_ptr6[static_cast<long>(x1 + (197L*x0))];
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = c10::convert<int>(x1);
                            auto tmp4 = static_cast<int>(1);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp5));
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = to_float_mask(tmp5);
                            auto tmp11 = at::vec::Vectorized<float>(tmp9);
                            auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                            auto tmp13 = tmp2 + tmp12;
                            auto tmp15 = tmp13 * tmp14;
                            auto tmp16 = [&]
                            {
                                auto tmp17 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp5));
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp16())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp16(), to_float_mask(tmp5));
                            auto tmp19 = decltype(tmp18)::blendv(tmp11, tmp18, tmp10);
                            auto tmp20 = tmp2 + tmp19;
                            auto tmp21 = tmp20 * tmp14;
                            auto tmp24 = at::vec::Vectorized<float>(tmp23);
                            auto tmp25 = tmp22 - tmp24;
                            auto tmp27 = at::vec::Vectorized<float>(tmp26);
                            auto tmp28 = tmp25 * tmp27;
                            auto tmp29 = tmp21 * tmp28;
                            tmp_acc0_vec = tmp_acc0_vec + tmp15;
                            tmp_acc1_vec = tmp_acc1_vec + tmp29;
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0);
                        tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp19 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        auto tmp23 = in_ptr5[static_cast<long>(x1 + (197L*x0))];
                        auto tmp26 = in_ptr6[static_cast<long>(x1 + (197L*x0))];
                        auto tmp29 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = c10::convert<int>(x1);
                        auto tmp4 = static_cast<int>(1);
                        auto tmp5 = tmp3 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = masked_load(in_ptr2 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp5));
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                        auto tmp9 = static_cast<float>(0.0);
                        auto tmp10 = to_float_mask(tmp5);
                        auto tmp11 = at::vec::Vectorized<float>(tmp9);
                        auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                        auto tmp13 = tmp2 + tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp16 = static_cast<float>(256.0);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 * tmp17;
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp22 - tmp24;
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp25 * tmp27;
                        auto tmp30 = at::vec::Vectorized<float>(tmp29);
                        auto tmp31 = tmp28 * tmp30;
                        auto tmp32 = tmp21 - tmp31;
                        auto tmp33 = tmp26 / tmp16;
                        auto tmp34 = at::vec::Vectorized<float>(tmp33);
                        auto tmp35 = tmp34 * tmp32;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = masked_load(in_ptr7 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp5));
                            return tmp37;
                        }
                        ;
                        auto tmp38 = decltype(tmp36())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp36(), to_float_mask(tmp5));
                        auto tmp39 = decltype(tmp38)::blendv(tmp11, tmp38, tmp10);
                        auto tmp40 = tmp35 + tmp39;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                        tmp40.store(out_ptr3 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_95 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = static_cast<float>(0.5);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp5 * tmp5;
                    auto tmp17 = static_cast<float>(-0.5);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp19.exp();
                    auto tmp21 = static_cast<float>(0.3989422804014327);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = tmp5 * tmp23;
                    auto tmp25 = tmp15 + tmp24;
                    auto tmp26 = tmp0 * tmp25;
                    auto tmp27 = tmp26 * tmp2;
                    auto tmp28 = tmp27 * tmp1;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp27;
                    tmp_acc1_vec = tmp_acc1_vec + tmp28;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(401L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp19 = in_ptr7[static_cast<long>(x1 + (401L*x0))];
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        auto tmp23 = in_ptr9[static_cast<long>(x1 + (401L*x0))];
                        auto tmp26 = in_ptr10[static_cast<long>(x1 + (401L*x0))];
                        auto tmp29 = in_ptr11[static_cast<long>(x1 + (401L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = c10::convert<int>(x1);
                        auto tmp4 = static_cast<int>(1);
                        auto tmp5 = tmp3 < tmp4;
                        auto tmp6 = [&]
                        {
                            auto tmp7 = masked_load(in_ptr5 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp5));
                            return tmp7;
                        }
                        ;
                        auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                        auto tmp9 = static_cast<float>(0.0);
                        auto tmp10 = to_float_mask(tmp5);
                        auto tmp11 = at::vec::Vectorized<float>(tmp9);
                        auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                        auto tmp13 = tmp2 + tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp16 = static_cast<float>(128.0);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 * tmp17;
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 - tmp20;
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp22 - tmp24;
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp25 * tmp27;
                        auto tmp30 = at::vec::Vectorized<float>(tmp29);
                        auto tmp31 = tmp28 * tmp30;
                        auto tmp32 = tmp21 - tmp31;
                        auto tmp33 = tmp26 / tmp16;
                        auto tmp34 = at::vec::Vectorized<float>(tmp33);
                        auto tmp35 = tmp34 * tmp32;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = masked_load(in_ptr12 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp5));
                            return tmp37;
                        }
                        ;
                        auto tmp38 = decltype(tmp36())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp36(), to_float_mask(tmp5));
                        auto tmp39 = decltype(tmp38)::blendv(tmp11, tmp38, tmp10);
                        auto tmp40 = tmp35 + tmp39;
                        auto tmp41 = tmp3 >= tmp4;
                        auto tmp42 = [&]
                        {
                            auto tmp43 = in_ptr10[static_cast<long>(x1 + (401L*x0))];
                            auto tmp44 = static_cast<float>(128.0);
                            auto tmp45 = tmp43 / tmp44;
                            auto tmp46 = at::vec::Vectorized<float>(tmp45);
                            auto tmp47 = tmp46 * tmp32;
                            auto tmp48 = c10::convert<int>(x1);
                            auto tmp49 = static_cast<int>(1);
                            auto tmp50 = tmp48 < tmp49;
                            auto tmp52 = tmp50 & tmp41;
                            auto tmp51 = [&]
                            {
                                auto tmp53 = masked_load(in_ptr12 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp52));
                                return tmp53;
                            }
                            ;
                            auto tmp54 = decltype(tmp51())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp51(), to_float_mask(tmp52));
                            auto tmp55 = static_cast<float>(0.0);
                            auto tmp56 = to_float_mask(tmp50);
                            auto tmp57 = at::vec::Vectorized<float>(tmp55);
                            auto tmp58 = decltype(tmp54)::blendv(tmp57, tmp54, tmp56);
                            auto tmp59 = tmp47 + tmp58;
                            return tmp59;
                        }
                        ;
                        auto tmp60 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp42(), to_float_mask(tmp41));
                        auto tmp61 = to_float_mask(tmp41);
                        auto tmp62 = decltype(tmp60)::blendv(tmp11, tmp60, tmp61);
                        auto tmp63 = [&]
                        {
                            auto tmp64 = masked_load(in_ptr13 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)), to_float_mask(tmp41));
                            return tmp64;
                        }
                        ;
                        auto tmp65 = decltype(tmp63())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp63(), to_float_mask(tmp41));
                        auto tmp66 = decltype(tmp65)::blendv(tmp11, tmp65, tmp61);
                        auto tmp67 = tmp62 + tmp66;
                        auto tmp68 = [&]
                        {
                            auto tmp69 = in_ptr14[static_cast<long>(x0)];
                            auto tmp70 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp5));
                            auto tmp71 = masked_load(in_ptr1 + static_cast<long>(x2), to_float_mask(tmp5));
                            auto tmp72 = tmp70 * tmp71;
                            auto tmp73 = static_cast<float>(128.0);
                            auto tmp74 = at::vec::Vectorized<float>(tmp73);
                            auto tmp75 = tmp72 * tmp74;
                            auto tmp76 = out_ptr0[static_cast<long>(x0)];
                            auto tmp77 = at::vec::Vectorized<float>(tmp76);
                            auto tmp78 = tmp75 - tmp77;
                            auto tmp79 = masked_load(in_ptr0 + static_cast<long>(x2 + (128L*x0)), to_float_mask(tmp5));
                            auto tmp80 = out_ptr1[static_cast<long>(x0)];
                            auto tmp81 = at::vec::Vectorized<float>(tmp80);
                            auto tmp82 = tmp79 * tmp81;
                            auto tmp83 = tmp78 - tmp82;
                            auto tmp84 = at::vec::Vectorized<float>(tmp69);
                            auto tmp85 = tmp84 * tmp83;
                            return tmp85;
                        }
                        ;
                        auto tmp86 = decltype(tmp68())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp68(), to_float_mask(tmp5));
                        auto tmp87 = decltype(tmp86)::blendv(tmp11, tmp86, tmp10);
                        auto tmp88 = tmp67 + tmp87;
                        tmp40.store(out_ptr3 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                        tmp88.store(out_ptr4 + static_cast<long>(x2 + (128L*x1) + (51328L*x0)));
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x2) + (51328L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (128L*x2) + (51328L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0 + (128L*x2) + (51328L*x1)));
                            auto tmp15 = in_ptr9[static_cast<long>(x2 + (401L*x1))];
                            auto tmp18 = in_ptr10[static_cast<long>(x2 + (401L*x1))];
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = static_cast<int>(1);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr5 + static_cast<long>(x0 + (128L*x1)), to_float_mask(tmp5));
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = to_float_mask(tmp5);
                            auto tmp11 = at::vec::Vectorized<float>(tmp9);
                            auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                            auto tmp13 = tmp2 + tmp12;
                            auto tmp16 = at::vec::Vectorized<float>(tmp15);
                            auto tmp17 = tmp14 - tmp16;
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp21 = tmp13 * tmp20;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = masked_load(in_ptr5 + static_cast<long>(x0 + (128L*x1)), to_float_mask(tmp5));
                                return tmp23;
                            }
                            ;
                            auto tmp24 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp5));
                            auto tmp25 = decltype(tmp24)::blendv(tmp11, tmp24, tmp10);
                            auto tmp26 = tmp2 + tmp25;
                            tmp_acc0_vec = tmp_acc0_vec + tmp21;
                            tmp_acc1_vec = tmp_acc1_vec + tmp26;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_backward_sum_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (51328L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_98 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_sum_99 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x2) + (50432L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x2) + (50432L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x2) + (50432L*x1)));
                            auto tmp15 = in_ptr4[static_cast<long>(x2 + (197L*x1))];
                            auto tmp18 = in_ptr5[static_cast<long>(x2 + (197L*x1))];
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = static_cast<int>(1);
                            auto tmp5 = tmp3 < tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr2 + static_cast<long>(x0 + (256L*x1)), to_float_mask(tmp5));
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            auto tmp9 = static_cast<float>(0.0);
                            auto tmp10 = to_float_mask(tmp5);
                            auto tmp11 = at::vec::Vectorized<float>(tmp9);
                            auto tmp12 = decltype(tmp8)::blendv(tmp11, tmp8, tmp10);
                            auto tmp13 = tmp2 + tmp12;
                            auto tmp16 = at::vec::Vectorized<float>(tmp15);
                            auto tmp17 = tmp14 - tmp16;
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 * tmp19;
                            auto tmp21 = tmp13 * tmp20;
                            auto tmp22 = [&]
                            {
                                auto tmp23 = masked_load(in_ptr2 + static_cast<long>(x0 + (256L*x1)), to_float_mask(tmp5));
                                return tmp23;
                            }
                            ;
                            auto tmp24 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp5));
                            auto tmp25 = decltype(tmp24)::blendv(tmp11, tmp24, tmp10);
                            auto tmp26 = tmp2 + tmp25;
                            tmp_acc0_vec = tmp_acc0_vec + tmp21;
                            tmp_acc1_vec = tmp_acc1_vec + tmp26;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_sum_101 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (51328L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = static_cast<float>(0.7071067811865476);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 * tmp7;
                    auto tmp9 = tmp8.erf();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = static_cast<float>(0.5);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp5 * tmp5;
                    auto tmp17 = static_cast<float>(-0.5);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp16 * tmp18;
                    auto tmp20 = tmp19.exp();
                    auto tmp21 = static_cast<float>(0.3989422804014327);
                    auto tmp22 = at::vec::Vectorized<float>(tmp21);
                    auto tmp23 = tmp20 * tmp22;
                    auto tmp24 = tmp5 * tmp23;
                    auto tmp25 = tmp15 + tmp24;
                    auto tmp26 = tmp0 * tmp25;
                    auto tmp27 = tmp26 * tmp2;
                    auto tmp28 = tmp27 * tmp1;
                    tmp26.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp27;
                    tmp_acc1_vec = tmp_acc1_vec + tmp28;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(1);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = masked_load(in_out_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp2));
                            return tmp4;
                        }
                        ;
                        auto tmp5 = decltype(tmp3())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp3(), to_float_mask(tmp2));
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = to_float_mask(tmp2);
                        auto tmp8 = at::vec::Vectorized<float>(tmp6);
                        auto tmp9 = decltype(tmp5)::blendv(tmp8, tmp5, tmp7);
                        auto tmp10 = [&]
                        {
                            auto tmp11 = in_ptr4[static_cast<long>(x1 + (197L*x0))];
                            auto tmp12 = static_cast<float>(256.0);
                            auto tmp13 = tmp11 / tmp12;
                            auto tmp14 = masked_load(in_ptr5 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)), to_float_mask(tmp2));
                            auto tmp15 = at::vec::Vectorized<float>(tmp13);
                            auto tmp16 = tmp15 * tmp14;
                            auto tmp17 = c10::convert<int>(x1);
                            auto tmp18 = static_cast<int>(1);
                            auto tmp19 = tmp17 < tmp18;
                            auto tmp21 = tmp19 & tmp2;
                            auto tmp20 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr6 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp21));
                                return tmp22;
                            }
                            ;
                            auto tmp23 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp21));
                            auto tmp24 = static_cast<float>(0.0);
                            auto tmp25 = to_float_mask(tmp19);
                            auto tmp26 = at::vec::Vectorized<float>(tmp24);
                            auto tmp27 = decltype(tmp23)::blendv(tmp26, tmp23, tmp25);
                            auto tmp28 = tmp16 + tmp27;
                            return tmp28;
                        }
                        ;
                        auto tmp29 = decltype(tmp10())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp10(), to_float_mask(tmp2));
                        auto tmp30 = decltype(tmp29)::blendv(tmp8, tmp29, tmp7);
                        auto tmp31 = tmp9 + tmp30;
                        auto tmp32 = tmp0 < tmp1;
                        auto tmp33 = [&]
                        {
                            auto tmp34 = in_ptr7[static_cast<long>(x0)];
                            auto tmp35 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp32));
                            auto tmp36 = masked_load(in_ptr2 + static_cast<long>(x2), to_float_mask(tmp32));
                            auto tmp37 = tmp35 * tmp36;
                            auto tmp38 = static_cast<float>(256.0);
                            auto tmp39 = at::vec::Vectorized<float>(tmp38);
                            auto tmp40 = tmp37 * tmp39;
                            auto tmp41 = out_ptr1[static_cast<long>(x0)];
                            auto tmp42 = at::vec::Vectorized<float>(tmp41);
                            auto tmp43 = tmp40 - tmp42;
                            auto tmp44 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*x0)), to_float_mask(tmp32));
                            auto tmp45 = out_ptr2[static_cast<long>(x0)];
                            auto tmp46 = at::vec::Vectorized<float>(tmp45);
                            auto tmp47 = tmp44 * tmp46;
                            auto tmp48 = tmp43 - tmp47;
                            auto tmp49 = at::vec::Vectorized<float>(tmp34);
                            auto tmp50 = tmp49 * tmp48;
                            return tmp50;
                        }
                        ;
                        auto tmp51 = decltype(tmp33())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp33(), to_float_mask(tmp32));
                        auto tmp52 = to_float_mask(tmp32);
                        auto tmp53 = decltype(tmp51)::blendv(tmp8, tmp51, tmp52);
                        auto tmp54 = tmp31 + tmp53;
                        tmp54.store(in_out_ptr1 + static_cast<long>(x2 + (256L*x1) + (50432L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_backward_sum_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (50432L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                    auto tmp2 = tmp0 * tmp1;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_103 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_104 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_105 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-403456L) + x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-806912L) + x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (256L*x0) + (768L*x2) + (151296L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_107 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_109 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-403456L) + x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-806912L) + x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (256L*x0) + (768L*x2) + (151296L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_112 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_113 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_114 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_115 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
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


cpp_fused_clone_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-403456L) + x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-806912L) + x3 + (256L*x2) + (50432L*x1) + (403456L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (256L*x0) + (768L*x2) + (151296L*x1)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_117 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1576L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(256.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_118 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1231872L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_119 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(128.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_sum_120 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_clone_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(401L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x3 + (128L*x2) + (51328L*x1) + (410624L*x0)), to_float_mask(tmp4));
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
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-410624L) + x3 + (128L*x2) + (51328L*x1) + (410624L*x0)), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(24);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>((-821248L) + x3 + (128L*x2) + (51328L*x1) + (410624L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            tmp24.store(out_ptr0 + static_cast<long>(x3 + (128L*x0) + (384L*x2) + (153984L*x1)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3208L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3208L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp12 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = static_cast<float>(128.0);
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
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50432L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (50432L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (50432L*x1)));
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


cpp_fused_sum_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(51328L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (51328L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
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
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (51328L*x1)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
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
    primals_5, primals_7, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_58, primals_61, primals_62, primals_65, primals_75, primals_76, primals_79, primals_89, primals_90, primals_93, primals_99, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_141, primals_142, primals_145, primals_146, primals_149, primals_159, primals_160, primals_163, primals_173, primals_174, primals_177, primals_183, primals_189, primals_195, primals_201, primals_207, primals_213, primals_219, primals_225, primals_226, primals_229, primals_230, primals_233, primals_243, primals_244, primals_247, primals_257, primals_258, primals_261, primals_263, primals_269, add_46, mul_82, view_6, getitem_2, getitem_3, getitem_4, getitem_6, getitem_7, getitem_8, getitem_11, getitem_12, view_10, mul_84, view_12, addmm_2, view_14, mul_89, view_16, getitem_18, getitem_19, getitem_20, getitem_22, getitem_23, getitem_24, getitem_27, getitem_28, view_20, mul_91, view_22, addmm_6, view_24, mul_96, view_26, getitem_34, getitem_35, getitem_36, getitem_38, getitem_39, getitem_40, getitem_43, getitem_44, view_30, mul_98, view_32, addmm_10, view_34, mul_103, view_36, getitem_50, getitem_51, getitem_52, getitem_54, getitem_55, getitem_56, getitem_59, getitem_60, view_40, mul_105, view_42, addmm_14, view_44, mul_110, view_46, mul_115, view_48, cat_2, getitem_69, rsqrt_10, view_50, view_53, view_66, mul_123, view_68, cat_3, cat_4, getitem_73, rsqrt_12, view_70, view_73, view_86, mul_131, view_88, cat_5, getitem_77, rsqrt_14, view_90, getitem_78, getitem_79, getitem_80, getitem_82, getitem_83, getitem_84, getitem_87, getitem_88, view_94, mul_138, view_96, addmm_28, view_98, getitem_93, rsqrt_16, view_100, getitem_94, getitem_95, getitem_96, getitem_98, getitem_99, getitem_100, getitem_103, getitem_104, view_104, mul_145, view_106, addmm_32, view_108, mul_150, view_110, getitem_110, getitem_111, getitem_112, getitem_114, getitem_115, getitem_116, getitem_119, getitem_120, view_114, mul_152, view_116, addmm_36, view_118, mul_157, view_120, getitem_126, getitem_127, getitem_128, getitem_130, getitem_131, getitem_132, getitem_135, getitem_136, view_124, mul_159, view_126, addmm_40, view_128, mul_164, view_130, mul_169, view_132, cat_6, getitem_145, rsqrt_24, view_134, view_137, view_150, mul_177, view_152, cat_7, cat_8, getitem_149, rsqrt_26, view_154, view_157, view_170, mul_185, view_172, cat_9, getitem_153, rsqrt_28, view_174, getitem_154, getitem_155, getitem_156, getitem_158, getitem_159, getitem_160, getitem_163, getitem_164, view_178, mul_192, view_180, addmm_54, view_182, getitem_169, rsqrt_30, view_184, getitem_170, getitem_171, getitem_172, getitem_174, getitem_175, getitem_176, getitem_179, getitem_180, view_188, mul_199, view_190, addmm_58, view_192, mul_204, view_194, getitem_186, getitem_187, getitem_188, getitem_190, getitem_191, getitem_192, getitem_195, getitem_196, view_198, mul_206, view_200, addmm_62, view_202, mul_211, view_204, getitem_202, getitem_203, getitem_204, getitem_206, getitem_207, getitem_208, getitem_211, getitem_212, view_208, mul_213, view_210, addmm_66, view_212, mul_218, view_214, mul_223, view_216, cat_10, getitem_221, rsqrt_38, view_218, view_221, view_234, mul_231, view_236, cat_11, cat_12, getitem_225, rsqrt_40, view_238, view_241, view_254, mul_239, view_256, cat_13, getitem_229, rsqrt_42, getitem_231, rsqrt_43, clone_68, clone_69, permute_142, permute_146, permute_150, div_9, permute_154, permute_159, permute_160, alias_18, permute_161, permute_162, permute_165, permute_170, permute_177, permute_179, div_11, permute_183, permute_188, permute_189, alias_19, permute_190, permute_191, permute_194, permute_199, permute_206, permute_208, div_13, permute_212, div_14, permute_216, permute_220, div_15, permute_224, alias_20, permute_230, div_16, permute_234, permute_238, div_17, permute_242, alias_21, permute_248, div_18, permute_252, permute_256, div_19, permute_260, alias_22, permute_266, permute_270, permute_274, div_21, permute_278, alias_23, permute_284, permute_288, div_23, permute_292, permute_297, permute_298, alias_24, permute_299, permute_300, permute_303, permute_308, permute_315, permute_317, div_25, permute_321, permute_326, permute_327, alias_25, permute_328, permute_329, permute_332, permute_337, permute_344, permute_346, div_27, permute_350, div_28, permute_354, permute_358, div_29, permute_362, alias_26, permute_368, div_30, permute_372, permute_376, div_31, permute_380, alias_27, permute_386, div_32, permute_390, permute_394, div_33, permute_398, alias_28, permute_404, permute_408, permute_412, div_35, permute_416, alias_29, permute_422, permute_426, div_37, permute_430, permute_435, permute_436, alias_30, permute_437, permute_438, permute_441, permute_446, permute_453, permute_455, div_39, permute_459, permute_464, permute_465, alias_31, permute_466, permute_467, permute_470, permute_475, permute_482, permute_484, div_41, permute_488, div_42, permute_492, permute_496, div_43, permute_500, alias_32, permute_506, div_44, permute_510, permute_514, div_45, permute_518, alias_33, permute_524, div_46, permute_528, permute_532, div_47, permute_536, alias_34, permute_542, div_48, permute_546, permute_550, div_49, permute_554, alias_35, permute_560, div_50, tangents_1 = args
    args.clear()
    assert_size_stride(primals_5, (128, 3, 12, 12), (432, 1, 36, 3))
    assert_size_stride(primals_7, (256, 3, 16, 16), (768, 1, 48, 3))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_21, (256, ), (1, ))
    assert_size_stride(primals_27, (256, ), (1, ))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_39, (256, ), (1, ))
    assert_size_stride(primals_45, (256, ), (1, ))
    assert_size_stride(primals_51, (256, ), (1, ))
    assert_size_stride(primals_57, (128, ), (1, ))
    assert_size_stride(primals_58, (128, ), (1, ))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_62, (256, ), (1, ))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, ), (1, ))
    assert_size_stride(primals_79, (128, ), (1, ))
    assert_size_stride(primals_89, (128, ), (1, ))
    assert_size_stride(primals_90, (128, ), (1, ))
    assert_size_stride(primals_93, (128, ), (1, ))
    assert_size_stride(primals_99, (128, ), (1, ))
    assert_size_stride(primals_105, (256, ), (1, ))
    assert_size_stride(primals_111, (256, ), (1, ))
    assert_size_stride(primals_117, (256, ), (1, ))
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_129, (256, ), (1, ))
    assert_size_stride(primals_135, (256, ), (1, ))
    assert_size_stride(primals_141, (128, ), (1, ))
    assert_size_stride(primals_142, (128, ), (1, ))
    assert_size_stride(primals_145, (256, ), (1, ))
    assert_size_stride(primals_146, (256, ), (1, ))
    assert_size_stride(primals_149, (256, ), (1, ))
    assert_size_stride(primals_159, (256, ), (1, ))
    assert_size_stride(primals_160, (256, ), (1, ))
    assert_size_stride(primals_163, (128, ), (1, ))
    assert_size_stride(primals_173, (128, ), (1, ))
    assert_size_stride(primals_174, (128, ), (1, ))
    assert_size_stride(primals_177, (128, ), (1, ))
    assert_size_stride(primals_183, (128, ), (1, ))
    assert_size_stride(primals_189, (256, ), (1, ))
    assert_size_stride(primals_195, (256, ), (1, ))
    assert_size_stride(primals_201, (256, ), (1, ))
    assert_size_stride(primals_207, (256, ), (1, ))
    assert_size_stride(primals_213, (256, ), (1, ))
    assert_size_stride(primals_219, (256, ), (1, ))
    assert_size_stride(primals_225, (128, ), (1, ))
    assert_size_stride(primals_226, (128, ), (1, ))
    assert_size_stride(primals_229, (256, ), (1, ))
    assert_size_stride(primals_230, (256, ), (1, ))
    assert_size_stride(primals_233, (256, ), (1, ))
    assert_size_stride(primals_243, (256, ), (1, ))
    assert_size_stride(primals_244, (256, ), (1, ))
    assert_size_stride(primals_247, (128, ), (1, ))
    assert_size_stride(primals_257, (128, ), (1, ))
    assert_size_stride(primals_258, (128, ), (1, ))
    assert_size_stride(primals_261, (128, ), (1, ))
    assert_size_stride(primals_263, (256, ), (1, ))
    assert_size_stride(primals_269, (8, 3, 240, 240), (172800, 1, 720, 3))
    assert_size_stride(add_46, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(mul_82, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(view_6, (3208, 128), (128, 1))
    assert_size_stride(getitem_2, (8, 4, 401, 32), (153984, 32, 384, 1))
    assert_size_stride(getitem_3, (8, 4, 401, 32), (153984, 32, 384, 1))
    assert_size_stride(getitem_4, (8, 4, 401, 32), (153984, 32, 384, 1))
    assert_size_stride(getitem_6, (8, 4, 401), (1604, 1, 4))
    assert_size_stride(getitem_7, (), ())
    assert_size_stride(getitem_8, (), ())
    assert_size_stride(getitem_11, (), ())
    assert_size_stride(getitem_12, (), ())
    assert_size_stride(view_10, (3208, 128), (128, 1))
    assert_size_stride(mul_84, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(view_12, (3208, 128), (128, 1))
    assert_size_stride(addmm_2, (3208, 384), (384, 1))
    assert_size_stride(view_14, (3208, 384), (384, 1))
    assert_size_stride(mul_89, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_16, (1576, 256), (256, 1))
    assert_size_stride(getitem_18, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_19, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_20, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_22, (8, 4, 197), (788, 1, 4))
    assert_size_stride(getitem_23, (), ())
    assert_size_stride(getitem_24, (), ())
    assert_size_stride(getitem_27, (), ())
    assert_size_stride(getitem_28, (), ())
    assert_size_stride(view_20, (1576, 256), (256, 1))
    assert_size_stride(mul_91, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_22, (1576, 256), (256, 1))
    assert_size_stride(addmm_6, (1576, 768), (768, 1))
    assert_size_stride(view_24, (1576, 768), (768, 1))
    assert_size_stride(mul_96, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_26, (1576, 256), (256, 1))
    assert_size_stride(getitem_34, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_35, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_36, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_38, (8, 4, 197), (788, 1, 4))
    assert_size_stride(getitem_39, (), ())
    assert_size_stride(getitem_40, (), ())
    assert_size_stride(getitem_43, (), ())
    assert_size_stride(getitem_44, (), ())
    assert_size_stride(view_30, (1576, 256), (256, 1))
    assert_size_stride(mul_98, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_32, (1576, 256), (256, 1))
    assert_size_stride(addmm_10, (1576, 768), (768, 1))
    assert_size_stride(view_34, (1576, 768), (768, 1))
    assert_size_stride(mul_103, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_36, (1576, 256), (256, 1))
    assert_size_stride(getitem_50, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_51, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_52, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_54, (8, 4, 197), (788, 1, 4))
    assert_size_stride(getitem_55, (), ())
    assert_size_stride(getitem_56, (), ())
    assert_size_stride(getitem_59, (), ())
    assert_size_stride(getitem_60, (), ())
    assert_size_stride(view_40, (1576, 256), (256, 1))
    assert_size_stride(mul_105, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_42, (1576, 256), (256, 1))
    assert_size_stride(addmm_14, (1576, 768), (768, 1))
    assert_size_stride(view_44, (1576, 768), (768, 1))
    assert_size_stride(mul_110, (8, 1, 128), (128, 128, 1))
    assert_size_stride(view_46, (8, 128), (128, 1))
    assert_size_stride(mul_115, (8, 1, 256), (256, 256, 1))
    assert_size_stride(view_48, (8, 256), (256, 1))
    assert_size_stride(cat_2, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(getitem_69, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_10, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_50, (8, 256), (50432, 1))
    assert_size_stride(view_53, (1576, 256), (256, 1))
    assert_size_stride(view_66, (8, 256), (256, 1))
    assert_size_stride(mul_123, (8, 1, 256), (256, 256, 1))
    assert_size_stride(view_68, (8, 256), (256, 1))
    assert_size_stride(cat_3, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(cat_4, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(getitem_73, (8, 401, 1), (401, 1, 1))
    assert_size_stride(rsqrt_12, (8, 401, 1), (401, 1, 1))
    assert_size_stride(view_70, (8, 128), (51328, 1))
    assert_size_stride(view_73, (3208, 128), (128, 1))
    assert_size_stride(view_86, (8, 128), (128, 1))
    assert_size_stride(mul_131, (8, 1, 128), (128, 128, 1))
    assert_size_stride(view_88, (8, 128), (128, 1))
    assert_size_stride(cat_5, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(getitem_77, (8, 401, 1), (401, 1, 1))
    assert_size_stride(rsqrt_14, (8, 401, 1), (401, 1, 1))
    assert_size_stride(view_90, (3208, 128), (128, 1))
    assert_size_stride(getitem_78, (8, 4, 401, 32), (153984, 32, 384, 1))
    assert_size_stride(getitem_79, (8, 4, 401, 32), (153984, 32, 384, 1))
    assert_size_stride(getitem_80, (8, 4, 401, 32), (153984, 32, 384, 1))
    assert_size_stride(getitem_82, (8, 4, 401), (1604, 1, 4))
    assert_size_stride(getitem_83, (), ())
    assert_size_stride(getitem_84, (), ())
    assert_size_stride(getitem_87, (), ())
    assert_size_stride(getitem_88, (), ())
    assert_size_stride(view_94, (3208, 128), (128, 1))
    assert_size_stride(mul_138, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(view_96, (3208, 128), (128, 1))
    assert_size_stride(addmm_28, (3208, 384), (384, 1))
    assert_size_stride(view_98, (3208, 384), (384, 1))
    assert_size_stride(getitem_93, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_16, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_100, (1576, 256), (256, 1))
    assert_size_stride(getitem_94, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_95, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_96, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_98, (8, 4, 197), (788, 1, 4))
    assert_size_stride(getitem_99, (), ())
    assert_size_stride(getitem_100, (), ())
    assert_size_stride(getitem_103, (), ())
    assert_size_stride(getitem_104, (), ())
    assert_size_stride(view_104, (1576, 256), (256, 1))
    assert_size_stride(mul_145, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_106, (1576, 256), (256, 1))
    assert_size_stride(addmm_32, (1576, 768), (768, 1))
    assert_size_stride(view_108, (1576, 768), (768, 1))
    assert_size_stride(mul_150, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_110, (1576, 256), (256, 1))
    assert_size_stride(getitem_110, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_111, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_112, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_114, (8, 4, 197), (788, 1, 4))
    assert_size_stride(getitem_115, (), ())
    assert_size_stride(getitem_116, (), ())
    assert_size_stride(getitem_119, (), ())
    assert_size_stride(getitem_120, (), ())
    assert_size_stride(view_114, (1576, 256), (256, 1))
    assert_size_stride(mul_152, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_116, (1576, 256), (256, 1))
    assert_size_stride(addmm_36, (1576, 768), (768, 1))
    assert_size_stride(view_118, (1576, 768), (768, 1))
    assert_size_stride(mul_157, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_120, (1576, 256), (256, 1))
    assert_size_stride(getitem_126, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_127, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_128, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_130, (8, 4, 197), (788, 1, 4))
    assert_size_stride(getitem_131, (), ())
    assert_size_stride(getitem_132, (), ())
    assert_size_stride(getitem_135, (), ())
    assert_size_stride(getitem_136, (), ())
    assert_size_stride(view_124, (1576, 256), (256, 1))
    assert_size_stride(mul_159, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_126, (1576, 256), (256, 1))
    assert_size_stride(addmm_40, (1576, 768), (768, 1))
    assert_size_stride(view_128, (1576, 768), (768, 1))
    assert_size_stride(mul_164, (8, 1, 128), (128, 128, 1))
    assert_size_stride(view_130, (8, 128), (128, 1))
    assert_size_stride(mul_169, (8, 1, 256), (256, 256, 1))
    assert_size_stride(view_132, (8, 256), (256, 1))
    assert_size_stride(cat_6, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(getitem_145, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_24, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_134, (8, 256), (50432, 1))
    assert_size_stride(view_137, (1576, 256), (256, 1))
    assert_size_stride(view_150, (8, 256), (256, 1))
    assert_size_stride(mul_177, (8, 1, 256), (256, 256, 1))
    assert_size_stride(view_152, (8, 256), (256, 1))
    assert_size_stride(cat_7, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(cat_8, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(getitem_149, (8, 401, 1), (401, 1, 1))
    assert_size_stride(rsqrt_26, (8, 401, 1), (401, 1, 1))
    assert_size_stride(view_154, (8, 128), (51328, 1))
    assert_size_stride(view_157, (3208, 128), (128, 1))
    assert_size_stride(view_170, (8, 128), (128, 1))
    assert_size_stride(mul_185, (8, 1, 128), (128, 128, 1))
    assert_size_stride(view_172, (8, 128), (128, 1))
    assert_size_stride(cat_9, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(getitem_153, (8, 401, 1), (401, 1, 1))
    assert_size_stride(rsqrt_28, (8, 401, 1), (401, 1, 1))
    assert_size_stride(view_174, (3208, 128), (128, 1))
    assert_size_stride(getitem_154, (8, 4, 401, 32), (153984, 32, 384, 1))
    assert_size_stride(getitem_155, (8, 4, 401, 32), (153984, 32, 384, 1))
    assert_size_stride(getitem_156, (8, 4, 401, 32), (153984, 32, 384, 1))
    assert_size_stride(getitem_158, (8, 4, 401), (1604, 1, 4))
    assert_size_stride(getitem_159, (), ())
    assert_size_stride(getitem_160, (), ())
    assert_size_stride(getitem_163, (), ())
    assert_size_stride(getitem_164, (), ())
    assert_size_stride(view_178, (3208, 128), (128, 1))
    assert_size_stride(mul_192, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(view_180, (3208, 128), (128, 1))
    assert_size_stride(addmm_54, (3208, 384), (384, 1))
    assert_size_stride(view_182, (3208, 384), (384, 1))
    assert_size_stride(getitem_169, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_30, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_184, (1576, 256), (256, 1))
    assert_size_stride(getitem_170, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_171, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_172, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_174, (8, 4, 197), (788, 1, 4))
    assert_size_stride(getitem_175, (), ())
    assert_size_stride(getitem_176, (), ())
    assert_size_stride(getitem_179, (), ())
    assert_size_stride(getitem_180, (), ())
    assert_size_stride(view_188, (1576, 256), (256, 1))
    assert_size_stride(mul_199, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_190, (1576, 256), (256, 1))
    assert_size_stride(addmm_58, (1576, 768), (768, 1))
    assert_size_stride(view_192, (1576, 768), (768, 1))
    assert_size_stride(mul_204, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_194, (1576, 256), (256, 1))
    assert_size_stride(getitem_186, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_187, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_188, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_190, (8, 4, 197), (788, 1, 4))
    assert_size_stride(getitem_191, (), ())
    assert_size_stride(getitem_192, (), ())
    assert_size_stride(getitem_195, (), ())
    assert_size_stride(getitem_196, (), ())
    assert_size_stride(view_198, (1576, 256), (256, 1))
    assert_size_stride(mul_206, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_200, (1576, 256), (256, 1))
    assert_size_stride(addmm_62, (1576, 768), (768, 1))
    assert_size_stride(view_202, (1576, 768), (768, 1))
    assert_size_stride(mul_211, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_204, (1576, 256), (256, 1))
    assert_size_stride(getitem_202, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_203, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_204, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_206, (8, 4, 197), (788, 1, 4))
    assert_size_stride(getitem_207, (), ())
    assert_size_stride(getitem_208, (), ())
    assert_size_stride(getitem_211, (), ())
    assert_size_stride(getitem_212, (), ())
    assert_size_stride(view_208, (1576, 256), (256, 1))
    assert_size_stride(mul_213, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_210, (1576, 256), (256, 1))
    assert_size_stride(addmm_66, (1576, 768), (768, 1))
    assert_size_stride(view_212, (1576, 768), (768, 1))
    assert_size_stride(mul_218, (8, 1, 128), (128, 128, 1))
    assert_size_stride(view_214, (8, 128), (128, 1))
    assert_size_stride(mul_223, (8, 1, 256), (256, 256, 1))
    assert_size_stride(view_216, (8, 256), (256, 1))
    assert_size_stride(cat_10, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(getitem_221, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_38, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_218, (8, 256), (50432, 1))
    assert_size_stride(view_221, (1576, 256), (256, 1))
    assert_size_stride(view_234, (8, 256), (256, 1))
    assert_size_stride(mul_231, (8, 1, 256), (256, 256, 1))
    assert_size_stride(view_236, (8, 256), (256, 1))
    assert_size_stride(cat_11, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(cat_12, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(getitem_225, (8, 401, 1), (401, 1, 1))
    assert_size_stride(rsqrt_40, (8, 401, 1), (401, 1, 1))
    assert_size_stride(view_238, (8, 128), (51328, 1))
    assert_size_stride(view_241, (3208, 128), (128, 1))
    assert_size_stride(view_254, (8, 128), (128, 1))
    assert_size_stride(mul_239, (8, 1, 128), (128, 128, 1))
    assert_size_stride(view_256, (8, 128), (128, 1))
    assert_size_stride(cat_13, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(getitem_229, (8, 401, 1), (401, 1, 1))
    assert_size_stride(rsqrt_42, (8, 401, 1), (401, 1, 1))
    assert_size_stride(getitem_231, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_43, (8, 197, 1), (197, 1, 1))
    assert_size_stride(clone_68, (8, 128), (128, 1))
    assert_size_stride(clone_69, (8, 256), (256, 1))
    assert_size_stride(permute_142, (1000, 256), (256, 1))
    assert_size_stride(permute_146, (1000, 128), (128, 1))
    assert_size_stride(permute_150, (256, 128), (128, 1))
    assert_size_stride(div_9, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_154, (128, 128), (128, 1))
    assert_size_stride(permute_159, (32, 401, 1), (401, 1, 0))
    assert_size_stride(permute_160, (32, 32, 401), (12832, 1, 32))
    assert_size_stride(alias_18, (8, 4, 1, 401), (1604, 1, 1604, 4))
    assert_size_stride(permute_161, (32, 32, 1), (32, 1, 0))
    assert_size_stride(permute_162, (32, 401, 32), (12832, 1, 401))
    assert_size_stride(permute_165, (128, 128), (128, 1))
    assert_size_stride(permute_170, (128, 128), (128, 1))
    assert_size_stride(permute_177, (128, 128), (128, 1))
    assert_size_stride(permute_179, (128, 256), (256, 1))
    assert_size_stride(div_11, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_183, (256, 256), (256, 1))
    assert_size_stride(permute_188, (32, 197, 1), (197, 1, 0))
    assert_size_stride(permute_189, (32, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_19, (8, 4, 1, 197), (788, 1, 788, 4))
    assert_size_stride(permute_190, (32, 64, 1), (64, 1, 0))
    assert_size_stride(permute_191, (32, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_194, (256, 256), (256, 1))
    assert_size_stride(permute_199, (256, 256), (256, 1))
    assert_size_stride(permute_206, (256, 256), (256, 1))
    assert_size_stride(permute_208, (128, 256), (256, 1))
    assert_size_stride(div_13, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_212, (256, 128), (128, 1))
    assert_size_stride(div_14, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_216, (256, 768), (768, 1))
    assert_size_stride(permute_220, (768, 256), (256, 1))
    assert_size_stride(div_15, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_224, (256, 256), (256, 1))
    assert_size_stride(alias_20, (8, 4, 197, 64), (50432, 1, 256, 4))
    assert_size_stride(permute_230, (768, 256), (256, 1))
    assert_size_stride(div_16, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_234, (256, 768), (768, 1))
    assert_size_stride(permute_238, (768, 256), (256, 1))
    assert_size_stride(div_17, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_242, (256, 256), (256, 1))
    assert_size_stride(alias_21, (8, 4, 197, 64), (50432, 1, 256, 4))
    assert_size_stride(permute_248, (768, 256), (256, 1))
    assert_size_stride(div_18, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_252, (256, 768), (768, 1))
    assert_size_stride(permute_256, (768, 256), (256, 1))
    assert_size_stride(div_19, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_260, (256, 256), (256, 1))
    assert_size_stride(alias_22, (8, 4, 197, 64), (50432, 1, 256, 4))
    assert_size_stride(permute_266, (768, 256), (256, 1))
    assert_size_stride(permute_270, (128, 384), (384, 1))
    assert_size_stride(permute_274, (384, 128), (128, 1))
    assert_size_stride(div_21, (8, 401, 1), (401, 1, 1))
    assert_size_stride(permute_278, (128, 128), (128, 1))
    assert_size_stride(alias_23, (8, 4, 401, 32), (51328, 1, 128, 4))
    assert_size_stride(permute_284, (384, 128), (128, 1))
    assert_size_stride(permute_288, (256, 128), (128, 1))
    assert_size_stride(div_23, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_292, (128, 128), (128, 1))
    assert_size_stride(permute_297, (32, 401, 1), (401, 1, 0))
    assert_size_stride(permute_298, (32, 32, 401), (12832, 1, 32))
    assert_size_stride(alias_24, (8, 4, 1, 401), (1604, 1, 1604, 4))
    assert_size_stride(permute_299, (32, 32, 1), (32, 1, 0))
    assert_size_stride(permute_300, (32, 401, 32), (12832, 1, 401))
    assert_size_stride(permute_303, (128, 128), (128, 1))
    assert_size_stride(permute_308, (128, 128), (128, 1))
    assert_size_stride(permute_315, (128, 128), (128, 1))
    assert_size_stride(permute_317, (128, 256), (256, 1))
    assert_size_stride(div_25, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_321, (256, 256), (256, 1))
    assert_size_stride(permute_326, (32, 197, 1), (197, 1, 0))
    assert_size_stride(permute_327, (32, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_25, (8, 4, 1, 197), (788, 1, 788, 4))
    assert_size_stride(permute_328, (32, 64, 1), (64, 1, 0))
    assert_size_stride(permute_329, (32, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_332, (256, 256), (256, 1))
    assert_size_stride(permute_337, (256, 256), (256, 1))
    assert_size_stride(permute_344, (256, 256), (256, 1))
    assert_size_stride(permute_346, (128, 256), (256, 1))
    assert_size_stride(div_27, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_350, (256, 128), (128, 1))
    assert_size_stride(div_28, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_354, (256, 768), (768, 1))
    assert_size_stride(permute_358, (768, 256), (256, 1))
    assert_size_stride(div_29, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_362, (256, 256), (256, 1))
    assert_size_stride(alias_26, (8, 4, 197, 64), (50432, 1, 256, 4))
    assert_size_stride(permute_368, (768, 256), (256, 1))
    assert_size_stride(div_30, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_372, (256, 768), (768, 1))
    assert_size_stride(permute_376, (768, 256), (256, 1))
    assert_size_stride(div_31, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_380, (256, 256), (256, 1))
    assert_size_stride(alias_27, (8, 4, 197, 64), (50432, 1, 256, 4))
    assert_size_stride(permute_386, (768, 256), (256, 1))
    assert_size_stride(div_32, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_390, (256, 768), (768, 1))
    assert_size_stride(permute_394, (768, 256), (256, 1))
    assert_size_stride(div_33, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_398, (256, 256), (256, 1))
    assert_size_stride(alias_28, (8, 4, 197, 64), (50432, 1, 256, 4))
    assert_size_stride(permute_404, (768, 256), (256, 1))
    assert_size_stride(permute_408, (128, 384), (384, 1))
    assert_size_stride(permute_412, (384, 128), (128, 1))
    assert_size_stride(div_35, (8, 401, 1), (401, 1, 1))
    assert_size_stride(permute_416, (128, 128), (128, 1))
    assert_size_stride(alias_29, (8, 4, 401, 32), (51328, 1, 128, 4))
    assert_size_stride(permute_422, (384, 128), (128, 1))
    assert_size_stride(permute_426, (256, 128), (128, 1))
    assert_size_stride(div_37, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_430, (128, 128), (128, 1))
    assert_size_stride(permute_435, (32, 401, 1), (401, 1, 0))
    assert_size_stride(permute_436, (32, 32, 401), (12832, 1, 32))
    assert_size_stride(alias_30, (8, 4, 1, 401), (1604, 1, 1604, 4))
    assert_size_stride(permute_437, (32, 32, 1), (32, 1, 0))
    assert_size_stride(permute_438, (32, 401, 32), (12832, 1, 401))
    assert_size_stride(permute_441, (128, 128), (128, 1))
    assert_size_stride(permute_446, (128, 128), (128, 1))
    assert_size_stride(permute_453, (128, 128), (128, 1))
    assert_size_stride(permute_455, (128, 256), (256, 1))
    assert_size_stride(div_39, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_459, (256, 256), (256, 1))
    assert_size_stride(permute_464, (32, 197, 1), (197, 1, 0))
    assert_size_stride(permute_465, (32, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_31, (8, 4, 1, 197), (788, 1, 788, 4))
    assert_size_stride(permute_466, (32, 64, 1), (64, 1, 0))
    assert_size_stride(permute_467, (32, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_470, (256, 256), (256, 1))
    assert_size_stride(permute_475, (256, 256), (256, 1))
    assert_size_stride(permute_482, (256, 256), (256, 1))
    assert_size_stride(permute_484, (128, 256), (256, 1))
    assert_size_stride(div_41, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_488, (256, 128), (128, 1))
    assert_size_stride(div_42, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_492, (256, 768), (768, 1))
    assert_size_stride(permute_496, (768, 256), (256, 1))
    assert_size_stride(div_43, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_500, (256, 256), (256, 1))
    assert_size_stride(alias_32, (8, 4, 197, 64), (50432, 1, 256, 4))
    assert_size_stride(permute_506, (768, 256), (256, 1))
    assert_size_stride(div_44, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_510, (256, 768), (768, 1))
    assert_size_stride(permute_514, (768, 256), (256, 1))
    assert_size_stride(div_45, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_518, (256, 256), (256, 1))
    assert_size_stride(alias_33, (8, 4, 197, 64), (50432, 1, 256, 4))
    assert_size_stride(permute_524, (768, 256), (256, 1))
    assert_size_stride(div_46, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_528, (256, 768), (768, 1))
    assert_size_stride(permute_532, (768, 256), (256, 1))
    assert_size_stride(div_47, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_536, (256, 256), (256, 1))
    assert_size_stride(alias_34, (8, 4, 197, 64), (50432, 1, 256, 4))
    assert_size_stride(permute_542, (768, 256), (256, 1))
    assert_size_stride(div_48, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_546, (128, 384), (384, 1))
    assert_size_stride(permute_550, (384, 128), (128, 1))
    assert_size_stride(div_49, (8, 401, 1), (401, 1, 1))
    assert_size_stride(permute_554, (128, 128), (128, 1))
    assert_size_stride(alias_35, (8, 4, 401, 32), (51328, 1, 128, 4))
    assert_size_stride(permute_560, (384, 128), (128, 1))
    assert_size_stride(div_50, (8, 401, 1), (401, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 1000), device='cpu', dtype=torch.float32)
    buf1 = empty((8, 1000), device='cpu', dtype=torch.float32)
    cpp_fused_div_select_0(c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del tangents_1
    buf2 = empty((8, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf1, permute_142, out=buf2)
    del permute_142
    buf3 = empty((1000, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf1, (1000, 8), (1, 1000), 0), clone_69, out=buf3)
    del clone_69
    buf4 = empty((1, 1000), device='cpu', dtype=torch.float32)
    cpp_fused_sum_1(c_void_p(buf1.data_ptr()), c_void_p(buf4.data_ptr()))
    del buf1
    buf5 = empty((8, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf0, permute_146, out=buf5)
    del permute_146
    buf6 = empty((1000, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf0, (1000, 8), (1, 1000), 0), clone_68, out=buf6)
    del clone_68
    buf7 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf9 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf10 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    buf11 = empty((256, ), device='cpu', dtype=torch.float32)
    buf12 = empty((256, ), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((8, 401, 1), (401, 1, 3208), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((8, 401, 1), (401, 1, 3208), device='cpu', dtype=torch.float32)
    buf15 = empty((8, 401, 128), device='cpu', dtype=torch.float32)
    buf16 = empty((128, ), device='cpu', dtype=torch.float32)
    buf17 = empty((128, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_select_backward_sum_2(c_void_p(buf0.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(cat_13.data_ptr()), c_void_p(getitem_231.data_ptr()), c_void_p(rsqrt_43.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(cat_11.data_ptr()), c_void_p(getitem_229.data_ptr()), c_void_p(rsqrt_42.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()))
    del buf0
    del cat_11
    del cat_13
    del getitem_229
    del getitem_231
    del primals_261
    del primals_263
    del rsqrt_42
    del rsqrt_43
    buf18 = buf5; del buf5  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf10, (8, 256), (50432, 1), 0), permute_150, out=buf18)
    del permute_150
    buf19 = empty((256, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf10, (256, 8), (1, 50432), 0), view_256, out=buf19)
    del view_256
    buf20 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf21 = reinterpret_tensor(buf18, (8, 1, 128), (128, 1024, 1), 0); del buf18  # reuse
    buf22 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf23 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf24 = empty((8, 1, 128), device='cpu', dtype=torch.float32)
    buf25 = empty((128, ), device='cpu', dtype=torch.float32)
    buf26 = empty((128, ), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_sum_3(c_void_p(buf21.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(mul_239.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(div_9.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()))
    del div_9
    del mul_239
    del primals_257
    del primals_258
    buf27 = reinterpret_tensor(buf21, (8, 128), (128, 1), 0); del buf21  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf24, (8, 128), (128, 1), 0), permute_154, out=buf27)
    del permute_154
    buf28 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf24, (128, 8), (1, 128), 0), view_254, out=buf28)
    del view_254
    buf29 = empty((1, 128), device='cpu', dtype=torch.float32)
    cpp_fused_sum_4(c_void_p(buf24.data_ptr()), c_void_p(buf29.data_ptr()))
    buf30 = empty((32, 401, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_159, reinterpret_tensor(buf27, (32, 1, 32), (32, 32, 1), 0), out=buf30)
    del permute_159
    buf31 = empty((32, 1, 401), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf27, (32, 1, 32), (32, 32, 1), 0), permute_160, out=buf31)
    del permute_160
    buf32 = empty_strided((8, 4, 1, 1), (4, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf33 = reinterpret_tensor(buf31, (8, 4, 1, 401), (1604, 401, 401, 1), 0); del buf31  # reuse
    cpp_fused__softmax_backward_data_mul_5(c_void_p(buf33.data_ptr()), c_void_p(alias_18.data_ptr()), c_void_p(buf32.data_ptr()))
    del alias_18
    buf34 = empty((32, 32, 401), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_161, reinterpret_tensor(buf33, (32, 1, 401), (401, 0, 1), 0), out=buf34)
    del permute_161
    buf35 = reinterpret_tensor(buf27, (32, 1, 32), (32, 32, 1), 0); del buf27  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf33, (32, 1, 401), (401, 0, 1), 0), permute_162, out=buf35)
    del permute_162
    buf36 = empty((3208, 128), device='cpu', dtype=torch.float32)
    cpp_fused_view_6(c_void_p(buf30.data_ptr()), c_void_p(buf36.data_ptr()))
    buf37 = reinterpret_tensor(buf30, (3208, 128), (128, 1), 0); del buf30  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf36, permute_165, out=buf37)
    del permute_165
    buf38 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf36, (128, 3208), (1, 128), 0), view_241, out=buf38)
    buf39 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf40 = empty((3208, 128), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_sum_7(c_void_p(buf36.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()))
    buf41 = buf36; del buf36  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf40, permute_170, out=buf41)
    del permute_170
    buf42 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf40, (128, 3208), (1, 128), 0), view_241, out=buf42)
    del view_241
    buf43 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf44 = empty((1, 1, 128), device='cpu', dtype=torch.float32)
    cpp_fused_sum_8(c_void_p(buf40.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()))
    buf45 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf35, (128, 8), (1, 128), 0), view_238, out=buf45)
    del view_238
    buf46 = empty((8, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf35, (8, 128), (128, 1), 0), permute_177, out=buf46)
    del permute_177
    buf47 = buf14; del buf14  # reuse
    buf48 = buf13; del buf13  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_9(c_void_p(buf37.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(cat_12.data_ptr()), c_void_p(getitem_225.data_ptr()), c_void_p(rsqrt_40.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    buf52 = buf2; del buf2  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf15, (8, 128), (51328, 1), 0), permute_179, out=buf52)
    del permute_179
    buf55 = reinterpret_tensor(buf52, (8, 1, 256), (256, 2048, 1), 0); del buf52  # reuse
    buf56 = buf23; del buf23  # reuse
    buf57 = buf22; del buf22  # reuse
    buf58 = empty((8, 1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_10(c_void_p(buf55.data_ptr()), c_void_p(mul_231.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(div_11.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()))
    del div_11
    del primals_243
    del primals_244
    buf61 = empty((8, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf58, (8, 256), (256, 1), 0), permute_183, out=buf61)
    del permute_183
    buf64 = empty((32, 197, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_188, reinterpret_tensor(buf61, (32, 1, 64), (64, 64, 1), 0), out=buf64)
    del permute_188
    buf70 = empty((1576, 256), device='cpu', dtype=torch.float32)
    cpp_fused_view_11(c_void_p(buf64.data_ptr()), c_void_p(buf70.data_ptr()))
    buf71 = reinterpret_tensor(buf64, (1576, 256), (256, 1), 0); del buf64  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf70, permute_194, out=buf71)
    del permute_194
    buf65 = empty((32, 1, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf61, (32, 1, 64), (64, 64, 1), 0), permute_189, out=buf65)
    del permute_189
    buf66 = buf32; del buf32  # reuse
    buf67 = reinterpret_tensor(buf65, (8, 4, 1, 197), (788, 197, 197, 1), 0); del buf65  # reuse
    cpp_fused__softmax_backward_data_mul_12(c_void_p(buf67.data_ptr()), c_void_p(alias_19.data_ptr()), c_void_p(buf66.data_ptr()))
    del alias_19
    buf68 = empty((32, 64, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_190, reinterpret_tensor(buf67, (32, 1, 197), (197, 0, 1), 0), out=buf68)
    del permute_190
    buf74 = empty((1576, 256), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_13(c_void_p(buf68.data_ptr()), c_void_p(buf74.data_ptr()))
    buf75 = reinterpret_tensor(buf68, (1576, 256), (256, 1), 0); del buf68  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf74, permute_199, out=buf75)
    del permute_199
    buf69 = reinterpret_tensor(buf61, (32, 1, 64), (64, 64, 1), 0); del buf61  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf67, (32, 1, 197), (197, 0, 1), 0), permute_191, out=buf69)
    del permute_191
    buf80 = empty((8, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf69, (8, 256), (256, 1), 0), permute_206, out=buf80)
    del permute_206
    buf81 = buf9; del buf9  # reuse
    buf82 = buf8; del buf8  # reuse
    buf83 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    buf96 = empty((8, 197, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_14(c_void_p(buf71.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(cat_10.data_ptr()), c_void_p(getitem_221.data_ptr()), c_void_p(rsqrt_38.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf96.data_ptr()))
    del primals_233
    buf97 = reinterpret_tensor(buf35, (8, 128), (128, 1), 0); del buf35  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf96, (8, 256), (50432, 1), 0), permute_212, out=buf97)
    del permute_212
    buf100 = reinterpret_tensor(buf97, (8, 1, 128), (128, 1024, 1), 0); del buf97  # reuse
    buf101 = buf57; del buf57  # reuse
    buf102 = buf56; del buf56  # reuse
    buf86 = reinterpret_tensor(buf40, (8, 401, 128), (51328, 128, 1), 0); del buf40  # reuse
    buf105 = reinterpret_tensor(buf34, (8, 401, 128), (51328, 128, 1), 0); del buf34  # reuse
    buf50 = empty((128, ), device='cpu', dtype=torch.float32)
    buf51 = empty((128, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_15(c_void_p(buf100.data_ptr()), c_void_p(mul_218.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(cat_12.data_ptr()), c_void_p(getitem_225.data_ptr()), c_void_p(rsqrt_40.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(div_14.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()))
    del buf37
    del buf41
    del cat_12
    del div_14
    del getitem_225
    del primals_225
    del primals_226
    del primals_247
    del rsqrt_40
    buf53 = empty((128, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf15, (128, 8), (1, 51328), 0), view_236, out=buf53)
    del view_236
    buf54 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf59 = empty((256, ), device='cpu', dtype=torch.float32)
    buf60 = empty((256, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_backward_sum_16(c_void_p(buf15.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(mul_231.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()))
    del buf15
    del mul_231
    buf62 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf58, (256, 8), (1, 256), 0), view_234, out=buf62)
    del view_234
    buf63 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_17(c_void_p(buf58.data_ptr()), c_void_p(buf63.data_ptr()))
    buf72 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf70, (256, 1576), (1, 256), 0), view_221, out=buf72)
    buf73 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_18(c_void_p(buf70.data_ptr()), c_void_p(buf73.data_ptr()))
    del buf70
    buf76 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf74, (256, 1576), (1, 256), 0), view_221, out=buf76)
    del view_221
    buf77 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf78 = empty((1, 1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_19(c_void_p(buf74.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()))
    del buf74
    buf79 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf69, (256, 8), (1, 256), 0), view_218, out=buf79)
    del view_218
    buf84 = empty((256, ), device='cpu', dtype=torch.float32)
    buf85 = empty((256, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_20(c_void_p(buf71.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(cat_10.data_ptr()), c_void_p(getitem_221.data_ptr()), c_void_p(rsqrt_38.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()))
    del buf71
    del buf75
    del cat_10
    del getitem_221
    buf87 = buf80; del buf80  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf86, (8, 128), (51328, 1), 0), permute_208, out=buf87)
    del permute_208
    buf88 = empty((128, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf86, (128, 8), (1, 51328), 0), view_216, out=buf88)
    del view_216
    buf89 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf90 = reinterpret_tensor(buf87, (8, 1, 256), (256, 2048, 1), 0); del buf87  # reuse
    buf91 = buf102; del buf102  # reuse
    buf92 = buf101; del buf101  # reuse
    buf93 = empty((256, ), device='cpu', dtype=torch.float32)
    buf94 = empty((256, ), device='cpu', dtype=torch.float32)
    buf95 = buf10; del buf10  # reuse
    cpp_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_sum_21(c_void_p(buf90.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(mul_223.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(rsqrt_38.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(div_13.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()))
    del buf83
    del div_13
    del mul_223
    del primals_229
    del primals_230
    del rsqrt_38
    buf98 = empty((256, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf96, (256, 8), (1, 50432), 0), view_214, out=buf98)
    del view_214
    buf99 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf103 = empty((128, ), device='cpu', dtype=torch.float32)
    buf104 = empty((128, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_backward_sum_22(c_void_p(buf96.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(mul_218.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()))
    del mul_218
    buf106 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf95, (1576, 256), (256, 1), 0), permute_216, out=buf106)
    del permute_216
    buf107 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf95, (256, 1576), (1, 256), 0), view_212, out=buf107)
    del view_212
    buf108 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf109 = reinterpret_tensor(buf106, (8, 197, 768), (151296, 768, 1), 0); del buf106  # reuse
    cpp_fused_gelu_gelu_backward_sum_23(c_void_p(buf109.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(addmm_66.data_ptr()), c_void_p(buf108.data_ptr()))
    del addmm_66
    buf110 = reinterpret_tensor(buf96, (1576, 256), (256, 1), 0); del buf96  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf109, (1576, 768), (768, 1), 0), permute_220, out=buf110)
    del permute_220
    buf111 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf109, (768, 1576), (1, 768), 0), view_210, out=buf111)
    del view_210
    buf112 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf113 = buf82; del buf82  # reuse
    buf114 = buf81; del buf81  # reuse
    buf115 = empty((256, ), device='cpu', dtype=torch.float32)
    buf116 = empty((256, ), device='cpu', dtype=torch.float32)
    buf117 = reinterpret_tensor(buf110, (8, 197, 256), (50432, 256, 1), 0); del buf110  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_24(c_void_p(buf117.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(mul_213.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(div_15.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()))
    del div_15
    del mul_213
    del primals_219
    buf118 = reinterpret_tensor(buf95, (1576, 256), (256, 1), 0); del buf95  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf117, (1576, 256), (256, 1), 0), permute_224, out=buf118)
    del permute_224
    buf119 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf117, (256, 1576), (1, 256), 0), view_208, out=buf119)
    del view_208
    buf120 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_25(c_void_p(buf117.data_ptr()), c_void_p(buf120.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf121 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf118, (8, 4, 197, 64), (50432, 64, 256, 1), 0), getitem_202, getitem_203, getitem_204, alias_20, getitem_206, getitem_207, getitem_208, 0, 0, 0.0, False, getitem_211, getitem_212)
    del alias_20
    del buf118
    del getitem_202
    del getitem_203
    del getitem_204
    del getitem_206
    del getitem_207
    del getitem_208
    del getitem_211
    del getitem_212
    buf122 = buf121[0]
    buf123 = buf121[1]
    buf124 = buf121[2]
    del buf121
    buf125 = reinterpret_tensor(buf109, (8, 197, 3, 4, 64), (151296, 768, 256, 64, 1), 0); del buf109  # reuse
    cpp_fused_clone_26(c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()))
    del buf122
    del buf123
    buf126 = reinterpret_tensor(buf124, (1576, 256), (256, 1), 0); del buf124  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf125, (1576, 768), (768, 1), 0), permute_230, out=buf126)
    del permute_230
    buf127 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf125, (768, 1576), (1, 768), 0), view_204, out=buf127)
    del view_204
    buf128 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf129 = buf114; del buf114  # reuse
    buf130 = buf113; del buf113  # reuse
    buf131 = empty((256, ), device='cpu', dtype=torch.float32)
    buf132 = empty((256, ), device='cpu', dtype=torch.float32)
    buf133 = buf117; del buf117  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_27(c_void_p(buf133.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(mul_211.data_ptr()), c_void_p(div_16.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    del div_16
    del mul_211
    del primals_213
    buf134 = reinterpret_tensor(buf125, (1576, 768), (768, 1), 0); del buf125  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf133, (1576, 256), (256, 1), 0), permute_234, out=buf134)
    del permute_234
    buf135 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf133, (256, 1576), (1, 256), 0), view_202, out=buf135)
    del view_202
    buf136 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf137 = reinterpret_tensor(buf134, (8, 197, 768), (151296, 768, 1), 0); del buf134  # reuse
    cpp_fused_gelu_gelu_backward_sum_28(c_void_p(buf137.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(addmm_62.data_ptr()), c_void_p(buf136.data_ptr()))
    del addmm_62
    buf138 = buf126; del buf126  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf137, (1576, 768), (768, 1), 0), permute_238, out=buf138)
    del permute_238
    buf139 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf137, (768, 1576), (1, 768), 0), view_200, out=buf139)
    del view_200
    buf140 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf141 = buf130; del buf130  # reuse
    buf142 = buf129; del buf129  # reuse
    buf143 = empty((256, ), device='cpu', dtype=torch.float32)
    buf144 = empty((256, ), device='cpu', dtype=torch.float32)
    buf145 = buf133; del buf133  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_29(c_void_p(buf145.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(mul_206.data_ptr()), c_void_p(div_17.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()))
    del div_17
    del mul_206
    del primals_207
    buf146 = buf138; del buf138  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (1576, 256), (256, 1), 0), permute_242, out=buf146)
    del permute_242
    buf147 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (256, 1576), (1, 256), 0), view_198, out=buf147)
    del view_198
    buf148 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_30(c_void_p(buf145.data_ptr()), c_void_p(buf148.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf149 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf146, (8, 4, 197, 64), (50432, 64, 256, 1), 0), getitem_186, getitem_187, getitem_188, alias_21, getitem_190, getitem_191, getitem_192, 0, 0, 0.0, False, getitem_195, getitem_196)
    del alias_21
    del buf146
    del getitem_186
    del getitem_187
    del getitem_188
    del getitem_190
    del getitem_191
    del getitem_192
    del getitem_195
    del getitem_196
    buf150 = buf149[0]
    buf151 = buf149[1]
    buf152 = buf149[2]
    del buf149
    buf153 = reinterpret_tensor(buf137, (8, 197, 3, 4, 64), (151296, 768, 256, 64, 1), 0); del buf137  # reuse
    cpp_fused_clone_31(c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()))
    buf154 = reinterpret_tensor(buf152, (1576, 256), (256, 1), 0); del buf152  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf153, (1576, 768), (768, 1), 0), permute_248, out=buf154)
    del permute_248
    buf155 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf153, (768, 1576), (1, 768), 0), view_194, out=buf155)
    del view_194
    buf156 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf157 = buf142; del buf142  # reuse
    buf158 = buf141; del buf141  # reuse
    buf159 = empty((256, ), device='cpu', dtype=torch.float32)
    buf160 = empty((256, ), device='cpu', dtype=torch.float32)
    buf161 = buf145; del buf145  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_32(c_void_p(buf161.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(mul_204.data_ptr()), c_void_p(div_18.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()))
    del div_18
    del mul_204
    del primals_201
    buf162 = reinterpret_tensor(buf153, (1576, 768), (768, 1), 0); del buf153  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf161, (1576, 256), (256, 1), 0), permute_252, out=buf162)
    del permute_252
    buf163 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf161, (256, 1576), (1, 256), 0), view_192, out=buf163)
    del view_192
    buf164 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf165 = reinterpret_tensor(buf162, (8, 197, 768), (151296, 768, 1), 0); del buf162  # reuse
    cpp_fused_gelu_gelu_backward_sum_33(c_void_p(buf165.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(addmm_58.data_ptr()), c_void_p(buf164.data_ptr()))
    del addmm_58
    buf166 = buf154; del buf154  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf165, (1576, 768), (768, 1), 0), permute_256, out=buf166)
    del permute_256
    buf167 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf165, (768, 1576), (1, 768), 0), view_190, out=buf167)
    del view_190
    buf168 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf169 = buf158; del buf158  # reuse
    buf170 = buf157; del buf157  # reuse
    buf171 = empty((256, ), device='cpu', dtype=torch.float32)
    buf172 = empty((256, ), device='cpu', dtype=torch.float32)
    buf173 = buf161; del buf161  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_34(c_void_p(buf173.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(mul_199.data_ptr()), c_void_p(div_19.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()))
    del div_19
    del mul_199
    del primals_195
    buf174 = buf166; del buf166  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf173, (1576, 256), (256, 1), 0), permute_260, out=buf174)
    del permute_260
    buf175 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf173, (256, 1576), (1, 256), 0), view_188, out=buf175)
    del view_188
    buf176 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_35(c_void_p(buf173.data_ptr()), c_void_p(buf176.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf177 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf174, (8, 4, 197, 64), (50432, 64, 256, 1), 0), getitem_170, getitem_171, getitem_172, alias_22, getitem_174, getitem_175, getitem_176, 0, 0, 0.0, False, getitem_179, getitem_180)
    del alias_22
    del getitem_170
    del getitem_171
    del getitem_172
    del getitem_174
    del getitem_175
    del getitem_176
    del getitem_179
    del getitem_180
    buf178 = buf177[0]
    buf179 = buf177[1]
    buf180 = buf177[2]
    del buf177
    buf181 = reinterpret_tensor(buf165, (8, 197, 3, 4, 64), (151296, 768, 256, 64, 1), 0); del buf165  # reuse
    cpp_fused_clone_36(c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()))
    buf182 = reinterpret_tensor(buf180, (1576, 256), (256, 1), 0); del buf180  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf181, (1576, 768), (768, 1), 0), permute_266, out=buf182)
    del permute_266
    buf183 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf181, (768, 1576), (1, 768), 0), view_184, out=buf183)
    del view_184
    buf184 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf185 = buf170; del buf170  # reuse
    buf186 = buf169; del buf169  # reuse
    buf187 = empty((256, ), device='cpu', dtype=torch.float32)
    buf188 = empty((256, ), device='cpu', dtype=torch.float32)
    buf189 = buf173; del buf173  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_37(c_void_p(buf189.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(cat_9.data_ptr()), c_void_p(getitem_169.data_ptr()), c_void_p(rsqrt_30.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()))
    del cat_9
    del getitem_169
    del primals_189
    del rsqrt_30
    buf190 = empty((3208, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf105, (3208, 128), (128, 1), 0), permute_270, out=buf190)
    del permute_270
    buf191 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf105, (128, 3208), (1, 128), 0), view_182, out=buf191)
    del view_182
    buf192 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf193 = reinterpret_tensor(buf190, (8, 401, 384), (153984, 384, 1), 0); del buf190  # reuse
    cpp_fused_gelu_gelu_backward_sum_38(c_void_p(buf193.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(addmm_54.data_ptr()), c_void_p(buf192.data_ptr()))
    del addmm_54
    buf194 = reinterpret_tensor(buf86, (3208, 128), (128, 1), 0); del buf86  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf193, (3208, 384), (384, 1), 0), permute_274, out=buf194)
    del permute_274
    buf195 = empty((384, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf193, (384, 3208), (1, 384), 0), view_180, out=buf195)
    del view_180
    buf196 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf197 = buf48; del buf48  # reuse
    buf198 = buf47; del buf47  # reuse
    buf199 = empty((128, ), device='cpu', dtype=torch.float32)
    buf200 = empty((128, ), device='cpu', dtype=torch.float32)
    buf201 = buf105; del buf105  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_39(c_void_p(buf201.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(mul_192.data_ptr()), c_void_p(div_21.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()))
    del div_21
    del mul_192
    del primals_183
    buf202 = buf194; del buf194  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf201, (3208, 128), (128, 1), 0), permute_278, out=buf202)
    del permute_278
    buf203 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf201, (128, 3208), (1, 128), 0), view_178, out=buf203)
    del view_178
    buf204 = empty((1, 128), device='cpu', dtype=torch.float32)
    cpp_fused_sum_40(c_void_p(buf201.data_ptr()), c_void_p(buf204.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf205 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf202, (8, 4, 401, 32), (51328, 32, 128, 1), 0), getitem_154, getitem_155, getitem_156, alias_23, getitem_158, getitem_159, getitem_160, 0, 0, 0.0, False, getitem_163, getitem_164)
    del alias_23
    del getitem_154
    del getitem_155
    del getitem_156
    del getitem_158
    del getitem_159
    del getitem_160
    del getitem_163
    del getitem_164
    buf206 = buf205[0]
    buf207 = buf205[1]
    buf208 = buf205[2]
    del buf205
    buf209 = reinterpret_tensor(buf193, (8, 401, 3, 4, 32), (153984, 384, 128, 32, 1), 0); del buf193  # reuse
    cpp_fused_clone_41(c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()))
    buf210 = reinterpret_tensor(buf208, (3208, 128), (128, 1), 0); del buf208  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf209, (3208, 384), (384, 1), 0), permute_284, out=buf210)
    del permute_284
    buf211 = empty((384, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf209, (384, 3208), (1, 384), 0), view_174, out=buf211)
    del view_174
    buf212 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf213 = buf198; del buf198  # reuse
    buf214 = buf197; del buf197  # reuse
    buf215 = empty((128, ), device='cpu', dtype=torch.float32)
    buf216 = empty((128, ), device='cpu', dtype=torch.float32)
    buf217 = buf201; del buf201  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_42(c_void_p(buf217.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(cat_7.data_ptr()), c_void_p(getitem_153.data_ptr()), c_void_p(rsqrt_28.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()))
    del cat_7
    del getitem_153
    del primals_177
    del rsqrt_28
    buf218 = reinterpret_tensor(buf100, (8, 128), (128, 1), 0); del buf100  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf189, (8, 256), (50432, 1), 0), permute_288, out=buf218)
    del permute_288
    buf219 = empty((256, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf189, (256, 8), (1, 50432), 0), view_172, out=buf219)
    del view_172
    buf220 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf221 = reinterpret_tensor(buf218, (8, 1, 128), (128, 1024, 1), 0); del buf218  # reuse
    buf222 = buf92; del buf92  # reuse
    buf223 = buf91; del buf91  # reuse
    buf224 = reinterpret_tensor(buf46, (8, 1, 128), (128, 128, 1), 0); del buf46  # reuse
    buf225 = empty((128, ), device='cpu', dtype=torch.float32)
    buf226 = empty((128, ), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_sum_43(c_void_p(buf221.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(mul_185.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(div_23.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()))
    del div_23
    del mul_185
    del primals_173
    del primals_174
    buf227 = reinterpret_tensor(buf221, (8, 128), (128, 1), 0); del buf221  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf224, (8, 128), (128, 1), 0), permute_292, out=buf227)
    del permute_292
    buf228 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf224, (128, 8), (1, 128), 0), view_170, out=buf228)
    del view_170
    buf229 = empty((1, 128), device='cpu', dtype=torch.float32)
    cpp_fused_sum_44(c_void_p(buf224.data_ptr()), c_void_p(buf229.data_ptr()))
    buf230 = reinterpret_tensor(buf210, (32, 401, 32), (12832, 32, 1), 0); del buf210  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_297, reinterpret_tensor(buf227, (32, 1, 32), (32, 32, 1), 0), out=buf230)
    del permute_297
    buf231 = reinterpret_tensor(buf33, (32, 1, 401), (401, 401, 1), 0); del buf33  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf227, (32, 1, 32), (32, 32, 1), 0), permute_298, out=buf231)
    del permute_298
    buf232 = buf66; del buf66  # reuse
    buf233 = reinterpret_tensor(buf231, (8, 4, 1, 401), (1604, 401, 401, 1), 0); del buf231  # reuse
    cpp_fused__softmax_backward_data_mul_45(c_void_p(buf233.data_ptr()), c_void_p(alias_24.data_ptr()), c_void_p(buf232.data_ptr()))
    del alias_24
    buf234 = reinterpret_tensor(buf207, (32, 32, 401), (12832, 401, 1), 0); del buf207  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_299, reinterpret_tensor(buf233, (32, 1, 401), (401, 0, 1), 0), out=buf234)
    del permute_299
    buf235 = reinterpret_tensor(buf227, (32, 1, 32), (32, 32, 1), 0); del buf227  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf233, (32, 1, 401), (401, 0, 1), 0), permute_300, out=buf235)
    del permute_300
    buf236 = reinterpret_tensor(buf206, (3208, 128), (128, 1), 0); del buf206  # reuse
    cpp_fused_view_46(c_void_p(buf230.data_ptr()), c_void_p(buf236.data_ptr()))
    buf237 = reinterpret_tensor(buf230, (3208, 128), (128, 1), 0); del buf230  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf236, permute_303, out=buf237)
    del permute_303
    buf238 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf236, (128, 3208), (1, 128), 0), view_157, out=buf238)
    buf239 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf240 = buf202; del buf202  # reuse
    cpp_fused__unsafe_view_clone_sum_47(c_void_p(buf236.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()))
    buf241 = buf236; del buf236  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf240, permute_308, out=buf241)
    del permute_308
    buf242 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf240, (128, 3208), (1, 128), 0), view_157, out=buf242)
    del view_157
    buf243 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf244 = empty((1, 1, 128), device='cpu', dtype=torch.float32)
    cpp_fused_sum_48(c_void_p(buf240.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()))
    buf245 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf235, (128, 8), (1, 128), 0), view_154, out=buf245)
    del view_154
    buf246 = reinterpret_tensor(buf24, (8, 128), (128, 1), 0); del buf24  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf235, (8, 128), (128, 1), 0), permute_315, out=buf246)
    del permute_315
    buf247 = buf214; del buf214  # reuse
    buf248 = buf213; del buf213  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_49(c_void_p(buf237.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(cat_8.data_ptr()), c_void_p(getitem_149.data_ptr()), c_void_p(rsqrt_26.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()))
    buf252 = reinterpret_tensor(buf90, (8, 256), (256, 1), 0); del buf90  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf217, (8, 128), (51328, 1), 0), permute_317, out=buf252)
    del permute_317
    buf255 = reinterpret_tensor(buf252, (8, 1, 256), (256, 2048, 1), 0); del buf252  # reuse
    buf256 = buf223; del buf223  # reuse
    buf257 = buf222; del buf222  # reuse
    buf258 = buf58; del buf58  # reuse
    cpp_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_50(c_void_p(buf255.data_ptr()), c_void_p(mul_177.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(div_25.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()))
    del div_25
    del primals_159
    del primals_160
    buf261 = reinterpret_tensor(buf69, (8, 256), (256, 1), 0); del buf69  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf258, (8, 256), (256, 1), 0), permute_321, out=buf261)
    del permute_321
    buf264 = reinterpret_tensor(buf182, (32, 197, 64), (12608, 64, 1), 0); del buf182  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_326, reinterpret_tensor(buf261, (32, 1, 64), (64, 64, 1), 0), out=buf264)
    del permute_326
    buf270 = reinterpret_tensor(buf179, (1576, 256), (256, 1), 0); del buf179  # reuse
    cpp_fused_view_51(c_void_p(buf264.data_ptr()), c_void_p(buf270.data_ptr()))
    buf271 = reinterpret_tensor(buf264, (1576, 256), (256, 1), 0); del buf264  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf270, permute_332, out=buf271)
    del permute_332
    buf265 = reinterpret_tensor(buf67, (32, 1, 197), (197, 197, 1), 0); del buf67  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf261, (32, 1, 64), (64, 64, 1), 0), permute_327, out=buf265)
    del permute_327
    buf266 = buf232; del buf232  # reuse
    buf267 = reinterpret_tensor(buf265, (8, 4, 1, 197), (788, 197, 197, 1), 0); del buf265  # reuse
    cpp_fused__softmax_backward_data_mul_52(c_void_p(buf267.data_ptr()), c_void_p(alias_25.data_ptr()), c_void_p(buf266.data_ptr()))
    del alias_25
    buf268 = reinterpret_tensor(buf178, (32, 64, 197), (12608, 197, 1), 0); del buf178  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_328, reinterpret_tensor(buf267, (32, 1, 197), (197, 0, 1), 0), out=buf268)
    del permute_328
    buf274 = buf174; del buf174  # reuse
    cpp_fused__unsafe_view_clone_53(c_void_p(buf268.data_ptr()), c_void_p(buf274.data_ptr()))
    buf275 = reinterpret_tensor(buf268, (1576, 256), (256, 1), 0); del buf268  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf274, permute_337, out=buf275)
    del permute_337
    buf269 = reinterpret_tensor(buf261, (32, 1, 64), (64, 64, 1), 0); del buf261  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf267, (32, 1, 197), (197, 0, 1), 0), permute_329, out=buf269)
    del permute_329
    buf280 = reinterpret_tensor(buf55, (8, 256), (256, 1), 0); del buf55  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf269, (8, 256), (256, 1), 0), permute_344, out=buf280)
    del permute_344
    buf281 = buf186; del buf186  # reuse
    buf282 = buf185; del buf185  # reuse
    buf283 = reinterpret_tensor(buf151, (8, 197, 256), (50432, 256, 1), 0); del buf151  # reuse
    buf296 = reinterpret_tensor(buf150, (8, 197, 256), (50432, 256, 1), 0); del buf150  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_54(c_void_p(buf271.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(cat_6.data_ptr()), c_void_p(getitem_145.data_ptr()), c_void_p(rsqrt_24.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf296.data_ptr()))
    del primals_149
    buf297 = reinterpret_tensor(buf235, (8, 128), (128, 1), 0); del buf235  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf296, (8, 256), (50432, 1), 0), permute_350, out=buf297)
    del permute_350
    buf300 = reinterpret_tensor(buf297, (8, 1, 128), (128, 1024, 1), 0); del buf297  # reuse
    buf301 = buf257; del buf257  # reuse
    buf302 = buf256; del buf256  # reuse
    buf286 = reinterpret_tensor(buf240, (8, 401, 128), (51328, 128, 1), 0); del buf240  # reuse
    buf305 = reinterpret_tensor(buf234, (8, 401, 128), (51328, 128, 1), 0); del buf234  # reuse
    buf250 = empty((128, ), device='cpu', dtype=torch.float32)
    buf251 = empty((128, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_55(c_void_p(buf300.data_ptr()), c_void_p(mul_164.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(cat_8.data_ptr()), c_void_p(getitem_149.data_ptr()), c_void_p(rsqrt_26.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(div_28.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()))
    del buf237
    del buf241
    del cat_8
    del div_28
    del getitem_149
    del primals_141
    del primals_142
    del primals_163
    del rsqrt_26
    buf253 = empty((128, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf217, (128, 8), (1, 51328), 0), view_152, out=buf253)
    del view_152
    buf254 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf259 = empty((256, ), device='cpu', dtype=torch.float32)
    buf260 = empty((256, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_backward_sum_56(c_void_p(buf217.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(mul_177.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()))
    del buf217
    del mul_177
    buf262 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf258, (256, 8), (1, 256), 0), view_150, out=buf262)
    del view_150
    buf263 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_57(c_void_p(buf258.data_ptr()), c_void_p(buf263.data_ptr()))
    buf272 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf270, (256, 1576), (1, 256), 0), view_137, out=buf272)
    buf273 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_58(c_void_p(buf270.data_ptr()), c_void_p(buf273.data_ptr()))
    del buf270
    buf276 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf274, (256, 1576), (1, 256), 0), view_137, out=buf276)
    del view_137
    buf277 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf278 = empty((1, 1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_59(c_void_p(buf274.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()))
    del buf274
    buf279 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf269, (256, 8), (1, 256), 0), view_134, out=buf279)
    del view_134
    buf284 = empty((256, ), device='cpu', dtype=torch.float32)
    buf285 = empty((256, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_60(c_void_p(buf271.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(cat_6.data_ptr()), c_void_p(getitem_145.data_ptr()), c_void_p(rsqrt_24.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()))
    del buf271
    del buf275
    del cat_6
    del getitem_145
    buf287 = buf280; del buf280  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf286, (8, 128), (51328, 1), 0), permute_346, out=buf287)
    del permute_346
    buf288 = empty((128, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf286, (128, 8), (1, 51328), 0), view_132, out=buf288)
    del view_132
    buf289 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf290 = reinterpret_tensor(buf287, (8, 1, 256), (256, 2048, 1), 0); del buf287  # reuse
    buf291 = buf302; del buf302  # reuse
    buf292 = buf301; del buf301  # reuse
    buf293 = empty((256, ), device='cpu', dtype=torch.float32)
    buf294 = empty((256, ), device='cpu', dtype=torch.float32)
    buf295 = buf189; del buf189  # reuse
    cpp_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_sum_61(c_void_p(buf290.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(mul_169.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(rsqrt_24.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(div_27.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()))
    del buf283
    del div_27
    del mul_169
    del primals_145
    del primals_146
    del rsqrt_24
    buf298 = empty((256, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf296, (256, 8), (1, 50432), 0), view_130, out=buf298)
    del view_130
    buf299 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf303 = empty((128, ), device='cpu', dtype=torch.float32)
    buf304 = empty((128, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_backward_sum_62(c_void_p(buf296.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(mul_164.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()))
    del mul_164
    buf306 = reinterpret_tensor(buf181, (1576, 768), (768, 1), 0); del buf181  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf295, (1576, 256), (256, 1), 0), permute_354, out=buf306)
    del permute_354
    buf307 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf295, (256, 1576), (1, 256), 0), view_128, out=buf307)
    del view_128
    buf308 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf309 = reinterpret_tensor(buf306, (8, 197, 768), (151296, 768, 1), 0); del buf306  # reuse
    cpp_fused_gelu_gelu_backward_sum_63(c_void_p(buf309.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(addmm_40.data_ptr()), c_void_p(buf308.data_ptr()))
    del addmm_40
    buf310 = reinterpret_tensor(buf296, (1576, 256), (256, 1), 0); del buf296  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf309, (1576, 768), (768, 1), 0), permute_358, out=buf310)
    del permute_358
    buf311 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf309, (768, 1576), (1, 768), 0), view_126, out=buf311)
    del view_126
    buf312 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf313 = buf282; del buf282  # reuse
    buf314 = buf281; del buf281  # reuse
    buf315 = empty((256, ), device='cpu', dtype=torch.float32)
    buf316 = empty((256, ), device='cpu', dtype=torch.float32)
    buf317 = buf295; del buf295  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_64(c_void_p(buf317.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(mul_159.data_ptr()), c_void_p(div_29.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()))
    del div_29
    del mul_159
    del primals_135
    buf318 = buf310; del buf310  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf317, (1576, 256), (256, 1), 0), permute_362, out=buf318)
    del permute_362
    buf319 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf317, (256, 1576), (1, 256), 0), view_124, out=buf319)
    del view_124
    buf320 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_65(c_void_p(buf317.data_ptr()), c_void_p(buf320.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf321 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf318, (8, 4, 197, 64), (50432, 64, 256, 1), 0), getitem_126, getitem_127, getitem_128, alias_26, getitem_130, getitem_131, getitem_132, 0, 0, 0.0, False, getitem_135, getitem_136)
    del alias_26
    del buf318
    del getitem_126
    del getitem_127
    del getitem_128
    del getitem_130
    del getitem_131
    del getitem_132
    del getitem_135
    del getitem_136
    buf322 = buf321[0]
    buf323 = buf321[1]
    buf324 = buf321[2]
    del buf321
    buf325 = reinterpret_tensor(buf309, (8, 197, 3, 4, 64), (151296, 768, 256, 64, 1), 0); del buf309  # reuse
    cpp_fused_clone_66(c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()))
    del buf322
    del buf323
    buf326 = reinterpret_tensor(buf324, (1576, 256), (256, 1), 0); del buf324  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf325, (1576, 768), (768, 1), 0), permute_368, out=buf326)
    del permute_368
    buf327 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf325, (768, 1576), (1, 768), 0), view_120, out=buf327)
    del view_120
    buf328 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf329 = buf314; del buf314  # reuse
    buf330 = buf313; del buf313  # reuse
    buf331 = empty((256, ), device='cpu', dtype=torch.float32)
    buf332 = empty((256, ), device='cpu', dtype=torch.float32)
    buf333 = buf317; del buf317  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_67(c_void_p(buf333.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(mul_157.data_ptr()), c_void_p(div_30.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()))
    del div_30
    del mul_157
    del primals_129
    buf334 = reinterpret_tensor(buf325, (1576, 768), (768, 1), 0); del buf325  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf333, (1576, 256), (256, 1), 0), permute_372, out=buf334)
    del permute_372
    buf335 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf333, (256, 1576), (1, 256), 0), view_118, out=buf335)
    del view_118
    buf336 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf337 = reinterpret_tensor(buf334, (8, 197, 768), (151296, 768, 1), 0); del buf334  # reuse
    cpp_fused_gelu_gelu_backward_sum_68(c_void_p(buf337.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(addmm_36.data_ptr()), c_void_p(buf336.data_ptr()))
    del addmm_36
    buf338 = buf326; del buf326  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf337, (1576, 768), (768, 1), 0), permute_376, out=buf338)
    del permute_376
    buf339 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf337, (768, 1576), (1, 768), 0), view_116, out=buf339)
    del view_116
    buf340 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf341 = buf330; del buf330  # reuse
    buf342 = buf329; del buf329  # reuse
    buf343 = empty((256, ), device='cpu', dtype=torch.float32)
    buf344 = empty((256, ), device='cpu', dtype=torch.float32)
    buf345 = buf333; del buf333  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_69(c_void_p(buf345.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(mul_152.data_ptr()), c_void_p(div_31.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()))
    del div_31
    del mul_152
    del primals_123
    buf346 = buf338; del buf338  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf345, (1576, 256), (256, 1), 0), permute_380, out=buf346)
    del permute_380
    buf347 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf345, (256, 1576), (1, 256), 0), view_114, out=buf347)
    del view_114
    buf348 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_70(c_void_p(buf345.data_ptr()), c_void_p(buf348.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf349 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf346, (8, 4, 197, 64), (50432, 64, 256, 1), 0), getitem_110, getitem_111, getitem_112, alias_27, getitem_114, getitem_115, getitem_116, 0, 0, 0.0, False, getitem_119, getitem_120)
    del alias_27
    del buf346
    del getitem_110
    del getitem_111
    del getitem_112
    del getitem_114
    del getitem_115
    del getitem_116
    del getitem_119
    del getitem_120
    buf350 = buf349[0]
    buf351 = buf349[1]
    buf352 = buf349[2]
    del buf349
    buf353 = reinterpret_tensor(buf337, (8, 197, 3, 4, 64), (151296, 768, 256, 64, 1), 0); del buf337  # reuse
    cpp_fused_clone_71(c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()))
    buf354 = reinterpret_tensor(buf352, (1576, 256), (256, 1), 0); del buf352  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf353, (1576, 768), (768, 1), 0), permute_386, out=buf354)
    del permute_386
    buf355 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf353, (768, 1576), (1, 768), 0), view_110, out=buf355)
    del view_110
    buf356 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf357 = buf342; del buf342  # reuse
    buf358 = buf341; del buf341  # reuse
    buf359 = empty((256, ), device='cpu', dtype=torch.float32)
    buf360 = empty((256, ), device='cpu', dtype=torch.float32)
    buf361 = buf345; del buf345  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_72(c_void_p(buf361.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(mul_150.data_ptr()), c_void_p(div_32.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()))
    del div_32
    del mul_150
    del primals_117
    buf362 = reinterpret_tensor(buf353, (1576, 768), (768, 1), 0); del buf353  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf361, (1576, 256), (256, 1), 0), permute_390, out=buf362)
    del permute_390
    buf363 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf361, (256, 1576), (1, 256), 0), view_108, out=buf363)
    del view_108
    buf364 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf365 = reinterpret_tensor(buf362, (8, 197, 768), (151296, 768, 1), 0); del buf362  # reuse
    cpp_fused_gelu_gelu_backward_sum_73(c_void_p(buf365.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(addmm_32.data_ptr()), c_void_p(buf364.data_ptr()))
    del addmm_32
    buf366 = buf354; del buf354  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf365, (1576, 768), (768, 1), 0), permute_394, out=buf366)
    del permute_394
    buf367 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf365, (768, 1576), (1, 768), 0), view_106, out=buf367)
    del view_106
    buf368 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf369 = buf358; del buf358  # reuse
    buf370 = buf357; del buf357  # reuse
    buf371 = empty((256, ), device='cpu', dtype=torch.float32)
    buf372 = empty((256, ), device='cpu', dtype=torch.float32)
    buf373 = buf361; del buf361  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_74(c_void_p(buf373.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(mul_145.data_ptr()), c_void_p(div_33.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()))
    del div_33
    del mul_145
    del primals_111
    buf374 = buf366; del buf366  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf373, (1576, 256), (256, 1), 0), permute_398, out=buf374)
    del permute_398
    buf375 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf373, (256, 1576), (1, 256), 0), view_104, out=buf375)
    del view_104
    buf376 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_75(c_void_p(buf373.data_ptr()), c_void_p(buf376.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf377 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf374, (8, 4, 197, 64), (50432, 64, 256, 1), 0), getitem_94, getitem_95, getitem_96, alias_28, getitem_98, getitem_99, getitem_100, 0, 0, 0.0, False, getitem_103, getitem_104)
    del alias_28
    del getitem_100
    del getitem_103
    del getitem_104
    del getitem_94
    del getitem_95
    del getitem_96
    del getitem_98
    del getitem_99
    buf378 = buf377[0]
    buf379 = buf377[1]
    buf380 = buf377[2]
    del buf377
    buf381 = reinterpret_tensor(buf365, (8, 197, 3, 4, 64), (151296, 768, 256, 64, 1), 0); del buf365  # reuse
    cpp_fused_clone_76(c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()))
    buf382 = reinterpret_tensor(buf380, (1576, 256), (256, 1), 0); del buf380  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf381, (1576, 768), (768, 1), 0), permute_404, out=buf382)
    del permute_404
    buf383 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf381, (768, 1576), (1, 768), 0), view_100, out=buf383)
    del view_100
    buf384 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf385 = buf370; del buf370  # reuse
    buf386 = buf369; del buf369  # reuse
    buf387 = empty((256, ), device='cpu', dtype=torch.float32)
    buf388 = empty((256, ), device='cpu', dtype=torch.float32)
    buf389 = buf373; del buf373  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_77(c_void_p(buf389.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(cat_5.data_ptr()), c_void_p(getitem_93.data_ptr()), c_void_p(rsqrt_16.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf388.data_ptr()))
    del cat_5
    del getitem_93
    del primals_105
    del rsqrt_16
    buf390 = reinterpret_tensor(buf209, (3208, 384), (384, 1), 0); del buf209  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf305, (3208, 128), (128, 1), 0), permute_408, out=buf390)
    del permute_408
    buf391 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf305, (128, 3208), (1, 128), 0), view_98, out=buf391)
    del view_98
    buf392 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf393 = reinterpret_tensor(buf390, (8, 401, 384), (153984, 384, 1), 0); del buf390  # reuse
    cpp_fused_gelu_gelu_backward_sum_78(c_void_p(buf393.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(addmm_28.data_ptr()), c_void_p(buf392.data_ptr()))
    del addmm_28
    buf394 = reinterpret_tensor(buf286, (3208, 128), (128, 1), 0); del buf286  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf393, (3208, 384), (384, 1), 0), permute_412, out=buf394)
    del permute_412
    buf395 = empty((384, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf393, (384, 3208), (1, 384), 0), view_96, out=buf395)
    del view_96
    buf396 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf397 = buf248; del buf248  # reuse
    buf398 = buf247; del buf247  # reuse
    buf399 = empty((128, ), device='cpu', dtype=torch.float32)
    buf400 = empty((128, ), device='cpu', dtype=torch.float32)
    buf401 = buf305; del buf305  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_79(c_void_p(buf401.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(mul_138.data_ptr()), c_void_p(div_35.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()))
    del div_35
    del mul_138
    del primals_99
    buf402 = buf394; del buf394  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf401, (3208, 128), (128, 1), 0), permute_416, out=buf402)
    del permute_416
    buf403 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf401, (128, 3208), (1, 128), 0), view_94, out=buf403)
    del view_94
    buf404 = empty((1, 128), device='cpu', dtype=torch.float32)
    cpp_fused_sum_80(c_void_p(buf401.data_ptr()), c_void_p(buf404.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf405 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf402, (8, 4, 401, 32), (51328, 32, 128, 1), 0), getitem_78, getitem_79, getitem_80, alias_29, getitem_82, getitem_83, getitem_84, 0, 0, 0.0, False, getitem_87, getitem_88)
    del alias_29
    del getitem_78
    del getitem_79
    del getitem_80
    del getitem_82
    del getitem_83
    del getitem_84
    del getitem_87
    del getitem_88
    buf406 = buf405[0]
    buf407 = buf405[1]
    buf408 = buf405[2]
    del buf405
    buf409 = reinterpret_tensor(buf393, (8, 401, 3, 4, 32), (153984, 384, 128, 32, 1), 0); del buf393  # reuse
    cpp_fused_clone_81(c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()))
    buf410 = reinterpret_tensor(buf408, (3208, 128), (128, 1), 0); del buf408  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf409, (3208, 384), (384, 1), 0), permute_422, out=buf410)
    del permute_422
    buf411 = empty((384, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf409, (384, 3208), (1, 384), 0), view_90, out=buf411)
    del view_90
    buf412 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf413 = buf398; del buf398  # reuse
    buf414 = buf397; del buf397  # reuse
    buf415 = empty((128, ), device='cpu', dtype=torch.float32)
    buf416 = empty((128, ), device='cpu', dtype=torch.float32)
    buf417 = buf401; del buf401  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_sum_82(c_void_p(buf417.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(cat_3.data_ptr()), c_void_p(getitem_77.data_ptr()), c_void_p(rsqrt_14.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()))
    del cat_3
    del getitem_77
    del primals_93
    del rsqrt_14
    buf418 = reinterpret_tensor(buf300, (8, 128), (128, 1), 0); del buf300  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf389, (8, 256), (50432, 1), 0), permute_426, out=buf418)
    del permute_426
    buf419 = empty((256, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf389, (256, 8), (1, 50432), 0), view_88, out=buf419)
    del view_88
    buf420 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf421 = reinterpret_tensor(buf418, (8, 1, 128), (128, 1024, 1), 0); del buf418  # reuse
    buf422 = buf292; del buf292  # reuse
    buf423 = buf291; del buf291  # reuse
    buf424 = reinterpret_tensor(buf246, (8, 1, 128), (128, 128, 1), 0); del buf246  # reuse
    buf425 = empty((128, ), device='cpu', dtype=torch.float32)
    buf426 = empty((128, ), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_sum_83(c_void_p(buf421.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(mul_131.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(div_37.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()))
    del div_37
    del mul_131
    del primals_89
    del primals_90
    buf427 = reinterpret_tensor(buf421, (8, 128), (128, 1), 0); del buf421  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf424, (8, 128), (128, 1), 0), permute_430, out=buf427)
    del permute_430
    buf428 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf424, (128, 8), (1, 128), 0), view_86, out=buf428)
    del view_86
    buf429 = empty((1, 128), device='cpu', dtype=torch.float32)
    cpp_fused_sum_84(c_void_p(buf424.data_ptr()), c_void_p(buf429.data_ptr()))
    buf430 = reinterpret_tensor(buf410, (32, 401, 32), (12832, 32, 1), 0); del buf410  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_435, reinterpret_tensor(buf427, (32, 1, 32), (32, 32, 1), 0), out=buf430)
    del permute_435
    buf431 = reinterpret_tensor(buf233, (32, 1, 401), (401, 401, 1), 0); del buf233  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf427, (32, 1, 32), (32, 32, 1), 0), permute_436, out=buf431)
    del permute_436
    buf432 = buf266; del buf266  # reuse
    buf433 = reinterpret_tensor(buf431, (8, 4, 1, 401), (1604, 401, 401, 1), 0); del buf431  # reuse
    cpp_fused__softmax_backward_data_mul_85(c_void_p(buf433.data_ptr()), c_void_p(alias_30.data_ptr()), c_void_p(buf432.data_ptr()))
    del alias_30
    buf434 = reinterpret_tensor(buf407, (32, 32, 401), (12832, 401, 1), 0); del buf407  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_437, reinterpret_tensor(buf433, (32, 1, 401), (401, 0, 1), 0), out=buf434)
    del permute_437
    buf435 = reinterpret_tensor(buf427, (32, 1, 32), (32, 32, 1), 0); del buf427  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf433, (32, 1, 401), (401, 0, 1), 0), permute_438, out=buf435)
    del buf433
    del permute_438
    buf436 = reinterpret_tensor(buf406, (3208, 128), (128, 1), 0); del buf406  # reuse
    cpp_fused_view_86(c_void_p(buf430.data_ptr()), c_void_p(buf436.data_ptr()))
    buf437 = reinterpret_tensor(buf430, (3208, 128), (128, 1), 0); del buf430  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf436, permute_441, out=buf437)
    del permute_441
    buf438 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf436, (128, 3208), (1, 128), 0), view_73, out=buf438)
    buf439 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf440 = buf402; del buf402  # reuse
    cpp_fused__unsafe_view_clone_sum_87(c_void_p(buf436.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()))
    buf441 = buf436; del buf436  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf440, permute_446, out=buf441)
    del permute_446
    buf442 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf440, (128, 3208), (1, 128), 0), view_73, out=buf442)
    del view_73
    buf443 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf444 = empty((1, 1, 128), device='cpu', dtype=torch.float32)
    cpp_fused_sum_88(c_void_p(buf440.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()))
    buf445 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf435, (128, 8), (1, 128), 0), view_70, out=buf445)
    del view_70
    buf446 = reinterpret_tensor(buf224, (8, 128), (128, 1), 0); del buf224  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf435, (8, 128), (128, 1), 0), permute_453, out=buf446)
    del permute_453
    buf447 = buf414; del buf414  # reuse
    buf448 = buf413; del buf413  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_89(c_void_p(buf437.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(cat_4.data_ptr()), c_void_p(getitem_73.data_ptr()), c_void_p(rsqrt_12.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()))
    buf452 = reinterpret_tensor(buf290, (8, 256), (256, 1), 0); del buf290  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf417, (8, 128), (51328, 1), 0), permute_455, out=buf452)
    del permute_455
    buf455 = reinterpret_tensor(buf452, (8, 1, 256), (256, 2048, 1), 0); del buf452  # reuse
    buf456 = buf423; del buf423  # reuse
    buf457 = buf422; del buf422  # reuse
    buf458 = buf258; del buf258  # reuse
    cpp_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_90(c_void_p(buf455.data_ptr()), c_void_p(mul_123.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(div_39.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()))
    del div_39
    del primals_75
    del primals_76
    buf461 = reinterpret_tensor(buf269, (8, 256), (256, 1), 0); del buf269  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf458, (8, 256), (256, 1), 0), permute_459, out=buf461)
    del permute_459
    buf464 = reinterpret_tensor(buf382, (32, 197, 64), (12608, 64, 1), 0); del buf382  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_464, reinterpret_tensor(buf461, (32, 1, 64), (64, 64, 1), 0), out=buf464)
    del permute_464
    buf470 = reinterpret_tensor(buf379, (1576, 256), (256, 1), 0); del buf379  # reuse
    cpp_fused_view_91(c_void_p(buf464.data_ptr()), c_void_p(buf470.data_ptr()))
    buf471 = reinterpret_tensor(buf464, (1576, 256), (256, 1), 0); del buf464  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf470, permute_470, out=buf471)
    del permute_470
    buf465 = reinterpret_tensor(buf267, (32, 1, 197), (197, 197, 1), 0); del buf267  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf461, (32, 1, 64), (64, 64, 1), 0), permute_465, out=buf465)
    del permute_465
    buf466 = buf432; del buf432  # reuse
    buf467 = reinterpret_tensor(buf465, (8, 4, 1, 197), (788, 197, 197, 1), 0); del buf465  # reuse
    cpp_fused__softmax_backward_data_mul_92(c_void_p(buf467.data_ptr()), c_void_p(alias_31.data_ptr()), c_void_p(buf466.data_ptr()))
    del alias_31
    del buf466
    buf468 = reinterpret_tensor(buf378, (32, 64, 197), (12608, 197, 1), 0); del buf378  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_466, reinterpret_tensor(buf467, (32, 1, 197), (197, 0, 1), 0), out=buf468)
    del permute_466
    buf474 = buf374; del buf374  # reuse
    cpp_fused__unsafe_view_clone_93(c_void_p(buf468.data_ptr()), c_void_p(buf474.data_ptr()))
    buf475 = reinterpret_tensor(buf468, (1576, 256), (256, 1), 0); del buf468  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf474, permute_475, out=buf475)
    del permute_475
    buf469 = reinterpret_tensor(buf461, (32, 1, 64), (64, 64, 1), 0); del buf461  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf467, (32, 1, 197), (197, 0, 1), 0), permute_467, out=buf469)
    del buf467
    del permute_467
    buf480 = reinterpret_tensor(buf255, (8, 256), (256, 1), 0); del buf255  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf469, (8, 256), (256, 1), 0), permute_482, out=buf480)
    del permute_482
    buf481 = buf386; del buf386  # reuse
    buf482 = buf385; del buf385  # reuse
    buf483 = reinterpret_tensor(buf351, (8, 197, 256), (50432, 256, 1), 0); del buf351  # reuse
    buf496 = reinterpret_tensor(buf350, (8, 197, 256), (50432, 256, 1), 0); del buf350  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_94(c_void_p(buf471.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(cat_2.data_ptr()), c_void_p(getitem_69.data_ptr()), c_void_p(rsqrt_10.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf496.data_ptr()))
    del primals_65
    buf497 = reinterpret_tensor(buf435, (8, 128), (128, 1), 0); del buf435  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf496, (8, 256), (50432, 1), 0), permute_488, out=buf497)
    del permute_488
    buf500 = reinterpret_tensor(buf497, (8, 1, 128), (128, 1024, 1), 0); del buf497  # reuse
    buf501 = buf457; del buf457  # reuse
    buf502 = buf456; del buf456  # reuse
    buf486 = reinterpret_tensor(buf440, (8, 401, 128), (51328, 128, 1), 0); del buf440  # reuse
    buf505 = reinterpret_tensor(buf434, (8, 401, 128), (51328, 128, 1), 0); del buf434  # reuse
    buf450 = empty((128, ), device='cpu', dtype=torch.float32)
    buf451 = empty((128, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_95(c_void_p(buf500.data_ptr()), c_void_p(mul_110.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(cat_4.data_ptr()), c_void_p(getitem_73.data_ptr()), c_void_p(rsqrt_12.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(div_42.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf451.data_ptr()))
    del buf424
    del buf437
    del buf441
    del buf446
    del cat_4
    del div_42
    del getitem_73
    del primals_57
    del primals_58
    del primals_79
    del rsqrt_12
    buf453 = empty((128, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf417, (128, 8), (1, 51328), 0), view_68, out=buf453)
    del view_68
    buf454 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf459 = empty((256, ), device='cpu', dtype=torch.float32)
    buf460 = empty((256, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_backward_sum_96(c_void_p(buf417.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(mul_123.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf460.data_ptr()))
    del buf417
    del buf455
    del mul_123
    buf462 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf458, (256, 8), (1, 256), 0), view_66, out=buf462)
    del view_66
    buf463 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_97(c_void_p(buf458.data_ptr()), c_void_p(buf463.data_ptr()))
    buf472 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf470, (256, 1576), (1, 256), 0), view_53, out=buf472)
    buf473 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_98(c_void_p(buf470.data_ptr()), c_void_p(buf473.data_ptr()))
    del buf470
    buf476 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf474, (256, 1576), (1, 256), 0), view_53, out=buf476)
    del view_53
    buf477 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf478 = empty((1, 1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_99(c_void_p(buf474.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()))
    del buf474
    buf479 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf469, (256, 8), (1, 256), 0), view_50, out=buf479)
    del buf469
    del view_50
    buf484 = empty((256, ), device='cpu', dtype=torch.float32)
    buf485 = empty((256, ), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_100(c_void_p(buf471.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(cat_2.data_ptr()), c_void_p(getitem_69.data_ptr()), c_void_p(rsqrt_10.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf485.data_ptr()))
    del buf471
    del buf475
    del cat_2
    del getitem_69
    buf487 = buf480; del buf480  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf486, (8, 128), (51328, 1), 0), permute_484, out=buf487)
    del permute_484
    buf488 = empty((128, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf486, (128, 8), (1, 51328), 0), view_48, out=buf488)
    del view_48
    buf489 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf490 = reinterpret_tensor(buf487, (8, 1, 256), (256, 2048, 1), 0); del buf487  # reuse
    buf491 = buf502; del buf502  # reuse
    buf492 = buf501; del buf501  # reuse
    buf493 = empty((256, ), device='cpu', dtype=torch.float32)
    buf494 = empty((256, ), device='cpu', dtype=torch.float32)
    buf495 = buf389; del buf389  # reuse
    cpp_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_sum_101(c_void_p(buf490.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(mul_115.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(rsqrt_10.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(div_41.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf494.data_ptr()))
    del buf458
    del buf483
    del buf490
    del buf491
    del buf492
    del div_41
    del mul_115
    del primals_61
    del primals_62
    del rsqrt_10
    buf498 = empty((256, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf496, (256, 8), (1, 50432), 0), view_46, out=buf498)
    del view_46
    buf499 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf503 = empty((128, ), device='cpu', dtype=torch.float32)
    buf504 = empty((128, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_backward_sum_102(c_void_p(buf496.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(mul_110.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()))
    del buf500
    del mul_110
    buf506 = reinterpret_tensor(buf381, (1576, 768), (768, 1), 0); del buf381  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf495, (1576, 256), (256, 1), 0), permute_492, out=buf506)
    del permute_492
    buf507 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf495, (256, 1576), (1, 256), 0), view_44, out=buf507)
    del view_44
    buf508 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf509 = reinterpret_tensor(buf506, (8, 197, 768), (151296, 768, 1), 0); del buf506  # reuse
    cpp_fused_gelu_gelu_backward_sum_103(c_void_p(buf509.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(addmm_14.data_ptr()), c_void_p(buf508.data_ptr()))
    del addmm_14
    buf510 = reinterpret_tensor(buf496, (1576, 256), (256, 1), 0); del buf496  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf509, (1576, 768), (768, 1), 0), permute_496, out=buf510)
    del permute_496
    buf511 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf509, (768, 1576), (1, 768), 0), view_42, out=buf511)
    del view_42
    buf512 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf513 = buf482; del buf482  # reuse
    buf514 = buf481; del buf481  # reuse
    buf515 = empty((256, ), device='cpu', dtype=torch.float32)
    buf516 = empty((256, ), device='cpu', dtype=torch.float32)
    buf517 = buf495; del buf495  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_104(c_void_p(buf517.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(mul_105.data_ptr()), c_void_p(div_43.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf516.data_ptr()))
    del div_43
    del mul_105
    del primals_51
    buf518 = buf510; del buf510  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf517, (1576, 256), (256, 1), 0), permute_500, out=buf518)
    del permute_500
    buf519 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf517, (256, 1576), (1, 256), 0), view_40, out=buf519)
    del view_40
    buf520 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_105(c_void_p(buf517.data_ptr()), c_void_p(buf520.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf521 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf518, (8, 4, 197, 64), (50432, 64, 256, 1), 0), getitem_50, getitem_51, getitem_52, alias_32, getitem_54, getitem_55, getitem_56, 0, 0, 0.0, False, getitem_59, getitem_60)
    del alias_32
    del buf518
    del getitem_50
    del getitem_51
    del getitem_52
    del getitem_54
    del getitem_55
    del getitem_56
    del getitem_59
    del getitem_60
    buf522 = buf521[0]
    buf523 = buf521[1]
    buf524 = buf521[2]
    del buf521
    buf525 = reinterpret_tensor(buf509, (8, 197, 3, 4, 64), (151296, 768, 256, 64, 1), 0); del buf509  # reuse
    cpp_fused_clone_106(c_void_p(buf522.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(buf525.data_ptr()))
    del buf522
    del buf523
    buf526 = reinterpret_tensor(buf524, (1576, 256), (256, 1), 0); del buf524  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf525, (1576, 768), (768, 1), 0), permute_506, out=buf526)
    del permute_506
    buf527 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf525, (768, 1576), (1, 768), 0), view_36, out=buf527)
    del view_36
    buf528 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf529 = buf514; del buf514  # reuse
    buf530 = buf513; del buf513  # reuse
    buf531 = empty((256, ), device='cpu', dtype=torch.float32)
    buf532 = empty((256, ), device='cpu', dtype=torch.float32)
    buf533 = buf517; del buf517  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_107(c_void_p(buf533.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(mul_103.data_ptr()), c_void_p(div_44.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()))
    del div_44
    del mul_103
    del primals_45
    buf534 = reinterpret_tensor(buf525, (1576, 768), (768, 1), 0); del buf525  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf533, (1576, 256), (256, 1), 0), permute_510, out=buf534)
    del permute_510
    buf535 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf533, (256, 1576), (1, 256), 0), view_34, out=buf535)
    del view_34
    buf536 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf537 = reinterpret_tensor(buf534, (8, 197, 768), (151296, 768, 1), 0); del buf534  # reuse
    cpp_fused_gelu_gelu_backward_sum_108(c_void_p(buf537.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(buf536.data_ptr()))
    del addmm_10
    buf538 = buf526; del buf526  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf537, (1576, 768), (768, 1), 0), permute_514, out=buf538)
    del permute_514
    buf539 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf537, (768, 1576), (1, 768), 0), view_32, out=buf539)
    del view_32
    buf540 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf541 = buf530; del buf530  # reuse
    buf542 = buf529; del buf529  # reuse
    buf543 = empty((256, ), device='cpu', dtype=torch.float32)
    buf544 = empty((256, ), device='cpu', dtype=torch.float32)
    buf545 = buf533; del buf533  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_109(c_void_p(buf545.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(mul_98.data_ptr()), c_void_p(div_45.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf544.data_ptr()))
    del div_45
    del mul_98
    del primals_39
    buf546 = buf538; del buf538  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf545, (1576, 256), (256, 1), 0), permute_518, out=buf546)
    del permute_518
    buf547 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf545, (256, 1576), (1, 256), 0), view_30, out=buf547)
    del view_30
    buf548 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_110(c_void_p(buf545.data_ptr()), c_void_p(buf548.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf549 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf546, (8, 4, 197, 64), (50432, 64, 256, 1), 0), getitem_34, getitem_35, getitem_36, alias_33, getitem_38, getitem_39, getitem_40, 0, 0, 0.0, False, getitem_43, getitem_44)
    del alias_33
    del buf546
    del getitem_34
    del getitem_35
    del getitem_36
    del getitem_38
    del getitem_39
    del getitem_40
    del getitem_43
    del getitem_44
    buf550 = buf549[0]
    buf551 = buf549[1]
    buf552 = buf549[2]
    del buf549
    buf553 = reinterpret_tensor(buf537, (8, 197, 3, 4, 64), (151296, 768, 256, 64, 1), 0); del buf537  # reuse
    cpp_fused_clone_111(c_void_p(buf550.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf553.data_ptr()))
    del buf550
    del buf551
    buf554 = reinterpret_tensor(buf552, (1576, 256), (256, 1), 0); del buf552  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf553, (1576, 768), (768, 1), 0), permute_524, out=buf554)
    del permute_524
    buf555 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf553, (768, 1576), (1, 768), 0), view_26, out=buf555)
    del view_26
    buf556 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf557 = buf542; del buf542  # reuse
    buf558 = buf541; del buf541  # reuse
    buf559 = empty((256, ), device='cpu', dtype=torch.float32)
    buf560 = empty((256, ), device='cpu', dtype=torch.float32)
    buf561 = buf545; del buf545  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_112(c_void_p(buf561.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(mul_96.data_ptr()), c_void_p(div_46.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf560.data_ptr()))
    del div_46
    del mul_96
    del primals_33
    buf562 = reinterpret_tensor(buf553, (1576, 768), (768, 1), 0); del buf553  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf561, (1576, 256), (256, 1), 0), permute_528, out=buf562)
    del permute_528
    buf563 = empty((256, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf561, (256, 1576), (1, 256), 0), view_24, out=buf563)
    del view_24
    buf564 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf565 = reinterpret_tensor(buf562, (8, 197, 768), (151296, 768, 1), 0); del buf562  # reuse
    cpp_fused_gelu_gelu_backward_sum_113(c_void_p(buf565.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(addmm_6.data_ptr()), c_void_p(buf564.data_ptr()))
    del addmm_6
    buf566 = buf554; del buf554  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf565, (1576, 768), (768, 1), 0), permute_532, out=buf566)
    del permute_532
    buf567 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf565, (768, 1576), (1, 768), 0), view_22, out=buf567)
    del view_22
    buf568 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf569 = buf558; del buf558  # reuse
    buf570 = buf557; del buf557  # reuse
    buf571 = empty((256, ), device='cpu', dtype=torch.float32)
    buf572 = empty((256, ), device='cpu', dtype=torch.float32)
    buf573 = buf561; del buf561  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_114(c_void_p(buf573.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(mul_91.data_ptr()), c_void_p(div_47.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(buf572.data_ptr()))
    del div_47
    del mul_91
    del primals_27
    buf574 = buf566; del buf566  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf573, (1576, 256), (256, 1), 0), permute_536, out=buf574)
    del permute_536
    buf575 = empty((256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf573, (256, 1576), (1, 256), 0), view_20, out=buf575)
    del view_20
    buf576 = empty((1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_sum_115(c_void_p(buf573.data_ptr()), c_void_p(buf576.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf577 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf574, (8, 4, 197, 64), (50432, 64, 256, 1), 0), getitem_18, getitem_19, getitem_20, alias_34, getitem_22, getitem_23, getitem_24, 0, 0, 0.0, False, getitem_27, getitem_28)
    del alias_34
    del buf574
    del getitem_18
    del getitem_19
    del getitem_20
    del getitem_22
    del getitem_23
    del getitem_24
    del getitem_27
    del getitem_28
    buf578 = buf577[0]
    buf579 = buf577[1]
    buf580 = buf577[2]
    del buf577
    buf581 = reinterpret_tensor(buf565, (8, 197, 3, 4, 64), (151296, 768, 256, 64, 1), 0); del buf565  # reuse
    cpp_fused_clone_116(c_void_p(buf578.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(buf581.data_ptr()))
    del buf578
    del buf579
    buf582 = reinterpret_tensor(buf580, (1576, 256), (256, 1), 0); del buf580  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf581, (1576, 768), (768, 1), 0), permute_542, out=buf582)
    del permute_542
    buf583 = empty((768, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf581, (768, 1576), (1, 768), 0), view_16, out=buf583)
    del view_16
    buf584 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf585 = buf570; del buf570  # reuse
    buf586 = buf569; del buf569  # reuse
    buf587 = empty((256, ), device='cpu', dtype=torch.float32)
    buf588 = empty((256, ), device='cpu', dtype=torch.float32)
    buf589 = buf573; del buf573  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_117(c_void_p(buf589.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf582.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(mul_89.data_ptr()), c_void_p(div_48.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf588.data_ptr()))
    del buf581
    del buf582
    del buf585
    del buf586
    del div_48
    del mul_89
    del primals_21
    buf590 = reinterpret_tensor(buf409, (3208, 384), (384, 1), 0); del buf409  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf505, (3208, 128), (128, 1), 0), permute_546, out=buf590)
    del permute_546
    buf591 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf505, (128, 3208), (1, 128), 0), view_14, out=buf591)
    del view_14
    buf592 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf593 = reinterpret_tensor(buf590, (8, 401, 384), (153984, 384, 1), 0); del buf590  # reuse
    cpp_fused_gelu_gelu_backward_sum_118(c_void_p(buf593.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(addmm_2.data_ptr()), c_void_p(buf592.data_ptr()))
    del addmm_2
    buf594 = reinterpret_tensor(buf486, (3208, 128), (128, 1), 0); del buf486  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf593, (3208, 384), (384, 1), 0), permute_550, out=buf594)
    del permute_550
    buf595 = empty((384, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf593, (384, 3208), (1, 384), 0), view_12, out=buf595)
    del view_12
    buf596 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf597 = buf448; del buf448  # reuse
    buf598 = buf447; del buf447  # reuse
    buf599 = empty((128, ), device='cpu', dtype=torch.float32)
    buf600 = empty((128, ), device='cpu', dtype=torch.float32)
    buf601 = buf505; del buf505  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_119(c_void_p(buf601.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(mul_84.data_ptr()), c_void_p(div_49.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(buf598.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf600.data_ptr()))
    del div_49
    del mul_84
    del primals_15
    buf602 = buf594; del buf594  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf601, (3208, 128), (128, 1), 0), permute_554, out=buf602)
    del permute_554
    buf603 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf601, (128, 3208), (1, 128), 0), view_10, out=buf603)
    del view_10
    buf604 = empty((1, 128), device='cpu', dtype=torch.float32)
    cpp_fused_sum_120(c_void_p(buf601.data_ptr()), c_void_p(buf604.data_ptr()))
    # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
    buf605 = aten._scaled_dot_product_flash_attention_backward(reinterpret_tensor(buf602, (8, 4, 401, 32), (51328, 32, 128, 1), 0), getitem_2, getitem_3, getitem_4, alias_35, getitem_6, getitem_7, getitem_8, 0, 0, 0.0, False, getitem_11, getitem_12)
    del alias_35
    del buf602
    del getitem_11
    del getitem_12
    del getitem_2
    del getitem_3
    del getitem_4
    del getitem_6
    del getitem_7
    del getitem_8
    buf606 = buf605[0]
    buf607 = buf605[1]
    buf608 = buf605[2]
    del buf605
    buf609 = reinterpret_tensor(buf593, (8, 401, 3, 4, 32), (153984, 384, 128, 32, 1), 0); del buf593  # reuse
    cpp_fused_clone_121(c_void_p(buf606.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf609.data_ptr()))
    del buf606
    del buf607
    buf610 = reinterpret_tensor(buf608, (3208, 128), (128, 1), 0); del buf608  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf609, (3208, 384), (384, 1), 0), permute_560, out=buf610)
    del permute_560
    buf611 = empty((384, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf609, (384, 3208), (1, 384), 0), view_6, out=buf611)
    del view_6
    buf612 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf613 = buf598; del buf598  # reuse
    buf614 = buf597; del buf597  # reuse
    buf615 = empty((128, ), device='cpu', dtype=torch.float32)
    buf616 = empty((128, ), device='cpu', dtype=torch.float32)
    buf617 = buf601; del buf601  # reuse
    buf618 = empty((1, 197, 256), device='cpu', dtype=torch.float32)
    buf619 = empty((1, 1, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_122(c_void_p(buf617.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(buf610.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(mul_82.data_ptr()), c_void_p(div_50.data_ptr()), c_void_p(buf589.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf615.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(buf618.data_ptr()), c_void_p(buf619.data_ptr()))
    del buf609
    del buf610
    del buf613
    del buf614
    del div_50
    del mul_82
    del primals_9
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf620 = aten.convolution_backward(reinterpret_tensor(buf589, (8, 256, 14, 14), (50432, 1, 3584, 256), 256), add_46, primals_7, [256], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True])
    del add_46
    del buf589
    del primals_7
    buf621 = buf620[1]
    buf622 = buf620[2]
    del buf620
    buf623 = empty((1, 401, 128), device='cpu', dtype=torch.float32)
    buf624 = empty((1, 1, 128), device='cpu', dtype=torch.float32)
    cpp_fused_sum_123(c_void_p(buf617.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(buf624.data_ptr()))
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf625 = aten.convolution_backward(reinterpret_tensor(buf617, (8, 128, 20, 20), (51328, 1, 2560, 128), 128), primals_269, primals_5, [128], [12, 12], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf617
    del primals_269
    del primals_5
    buf626 = buf625[1]
    buf627 = buf625[2]
    return (buf624, buf623, buf619, buf618, buf626, buf627, buf621, buf622, buf615, buf616, reinterpret_tensor(buf611, (384, 128), (128, 1), 0), reinterpret_tensor(buf612, (384, ), (1, ), 0), reinterpret_tensor(buf603, (128, 128), (128, 1), 0), reinterpret_tensor(buf604, (128, ), (1, ), 0), buf599, buf600, reinterpret_tensor(buf595, (384, 128), (128, 1), 0), reinterpret_tensor(buf596, (384, ), (1, ), 0), reinterpret_tensor(buf591, (128, 384), (384, 1), 0), reinterpret_tensor(buf592, (128, ), (1, ), 0), buf587, buf588, reinterpret_tensor(buf583, (768, 256), (256, 1), 0), reinterpret_tensor(buf584, (768, ), (1, ), 0), reinterpret_tensor(buf575, (256, 256), (256, 1), 0), reinterpret_tensor(buf576, (256, ), (1, ), 0), buf571, buf572, reinterpret_tensor(buf567, (768, 256), (256, 1), 0), reinterpret_tensor(buf568, (768, ), (1, ), 0), reinterpret_tensor(buf563, (256, 768), (768, 1), 0), reinterpret_tensor(buf564, (256, ), (1, ), 0), buf559, buf560, reinterpret_tensor(buf555, (768, 256), (256, 1), 0), reinterpret_tensor(buf556, (768, ), (1, ), 0), reinterpret_tensor(buf547, (256, 256), (256, 1), 0), reinterpret_tensor(buf548, (256, ), (1, ), 0), buf543, buf544, reinterpret_tensor(buf539, (768, 256), (256, 1), 0), reinterpret_tensor(buf540, (768, ), (1, ), 0), reinterpret_tensor(buf535, (256, 768), (768, 1), 0), reinterpret_tensor(buf536, (256, ), (1, ), 0), buf531, buf532, reinterpret_tensor(buf527, (768, 256), (256, 1), 0), reinterpret_tensor(buf528, (768, ), (1, ), 0), reinterpret_tensor(buf519, (256, 256), (256, 1), 0), reinterpret_tensor(buf520, (256, ), (1, ), 0), buf515, buf516, reinterpret_tensor(buf511, (768, 256), (256, 1), 0), reinterpret_tensor(buf512, (768, ), (1, ), 0), reinterpret_tensor(buf507, (256, 768), (768, 1), 0), reinterpret_tensor(buf508, (256, ), (1, ), 0), buf503, buf504, reinterpret_tensor(buf498, (256, 128), (128, 1), 0), reinterpret_tensor(buf499, (256, ), (1, ), 0), buf493, buf494, reinterpret_tensor(buf488, (128, 256), (256, 1), 0), reinterpret_tensor(buf489, (128, ), (1, ), 0), buf484, buf485, reinterpret_tensor(buf479, (256, 256), (256, 1), 0), reinterpret_tensor(buf478, (256, ), (1, ), 0), reinterpret_tensor(buf476, (256, 256), (256, 1), 0), reinterpret_tensor(buf477, (256, ), (1, ), 0), reinterpret_tensor(buf472, (256, 256), (256, 1), 0), reinterpret_tensor(buf473, (256, ), (1, ), 0), reinterpret_tensor(buf462, (256, 256), (256, 1), 0), reinterpret_tensor(buf463, (256, ), (1, ), 0), buf459, buf460, reinterpret_tensor(buf453, (128, 256), (256, 1), 0), reinterpret_tensor(buf454, (128, ), (1, ), 0), buf450, buf451, reinterpret_tensor(buf445, (128, 128), (128, 1), 0), reinterpret_tensor(buf444, (128, ), (1, ), 0), reinterpret_tensor(buf442, (128, 128), (128, 1), 0), reinterpret_tensor(buf443, (128, ), (1, ), 0), reinterpret_tensor(buf438, (128, 128), (128, 1), 0), reinterpret_tensor(buf439, (128, ), (1, ), 0), reinterpret_tensor(buf428, (128, 128), (128, 1), 0), reinterpret_tensor(buf429, (128, ), (1, ), 0), buf425, buf426, reinterpret_tensor(buf419, (256, 128), (128, 1), 0), reinterpret_tensor(buf420, (256, ), (1, ), 0), buf415, buf416, reinterpret_tensor(buf411, (384, 128), (128, 1), 0), reinterpret_tensor(buf412, (384, ), (1, ), 0), reinterpret_tensor(buf403, (128, 128), (128, 1), 0), reinterpret_tensor(buf404, (128, ), (1, ), 0), buf399, buf400, reinterpret_tensor(buf395, (384, 128), (128, 1), 0), reinterpret_tensor(buf396, (384, ), (1, ), 0), reinterpret_tensor(buf391, (128, 384), (384, 1), 0), reinterpret_tensor(buf392, (128, ), (1, ), 0), buf387, buf388, reinterpret_tensor(buf383, (768, 256), (256, 1), 0), reinterpret_tensor(buf384, (768, ), (1, ), 0), reinterpret_tensor(buf375, (256, 256), (256, 1), 0), reinterpret_tensor(buf376, (256, ), (1, ), 0), buf371, buf372, reinterpret_tensor(buf367, (768, 256), (256, 1), 0), reinterpret_tensor(buf368, (768, ), (1, ), 0), reinterpret_tensor(buf363, (256, 768), (768, 1), 0), reinterpret_tensor(buf364, (256, ), (1, ), 0), buf359, buf360, reinterpret_tensor(buf355, (768, 256), (256, 1), 0), reinterpret_tensor(buf356, (768, ), (1, ), 0), reinterpret_tensor(buf347, (256, 256), (256, 1), 0), reinterpret_tensor(buf348, (256, ), (1, ), 0), buf343, buf344, reinterpret_tensor(buf339, (768, 256), (256, 1), 0), reinterpret_tensor(buf340, (768, ), (1, ), 0), reinterpret_tensor(buf335, (256, 768), (768, 1), 0), reinterpret_tensor(buf336, (256, ), (1, ), 0), buf331, buf332, reinterpret_tensor(buf327, (768, 256), (256, 1), 0), reinterpret_tensor(buf328, (768, ), (1, ), 0), reinterpret_tensor(buf319, (256, 256), (256, 1), 0), reinterpret_tensor(buf320, (256, ), (1, ), 0), buf315, buf316, reinterpret_tensor(buf311, (768, 256), (256, 1), 0), reinterpret_tensor(buf312, (768, ), (1, ), 0), reinterpret_tensor(buf307, (256, 768), (768, 1), 0), reinterpret_tensor(buf308, (256, ), (1, ), 0), buf303, buf304, reinterpret_tensor(buf298, (256, 128), (128, 1), 0), reinterpret_tensor(buf299, (256, ), (1, ), 0), buf293, buf294, reinterpret_tensor(buf288, (128, 256), (256, 1), 0), reinterpret_tensor(buf289, (128, ), (1, ), 0), buf284, buf285, reinterpret_tensor(buf279, (256, 256), (256, 1), 0), reinterpret_tensor(buf278, (256, ), (1, ), 0), reinterpret_tensor(buf276, (256, 256), (256, 1), 0), reinterpret_tensor(buf277, (256, ), (1, ), 0), reinterpret_tensor(buf272, (256, 256), (256, 1), 0), reinterpret_tensor(buf273, (256, ), (1, ), 0), reinterpret_tensor(buf262, (256, 256), (256, 1), 0), reinterpret_tensor(buf263, (256, ), (1, ), 0), buf259, buf260, reinterpret_tensor(buf253, (128, 256), (256, 1), 0), reinterpret_tensor(buf254, (128, ), (1, ), 0), buf250, buf251, reinterpret_tensor(buf245, (128, 128), (128, 1), 0), reinterpret_tensor(buf244, (128, ), (1, ), 0), reinterpret_tensor(buf242, (128, 128), (128, 1), 0), reinterpret_tensor(buf243, (128, ), (1, ), 0), reinterpret_tensor(buf238, (128, 128), (128, 1), 0), reinterpret_tensor(buf239, (128, ), (1, ), 0), reinterpret_tensor(buf228, (128, 128), (128, 1), 0), reinterpret_tensor(buf229, (128, ), (1, ), 0), buf225, buf226, reinterpret_tensor(buf219, (256, 128), (128, 1), 0), reinterpret_tensor(buf220, (256, ), (1, ), 0), buf215, buf216, reinterpret_tensor(buf211, (384, 128), (128, 1), 0), reinterpret_tensor(buf212, (384, ), (1, ), 0), reinterpret_tensor(buf203, (128, 128), (128, 1), 0), reinterpret_tensor(buf204, (128, ), (1, ), 0), buf199, buf200, reinterpret_tensor(buf195, (384, 128), (128, 1), 0), reinterpret_tensor(buf196, (384, ), (1, ), 0), reinterpret_tensor(buf191, (128, 384), (384, 1), 0), reinterpret_tensor(buf192, (128, ), (1, ), 0), buf187, buf188, reinterpret_tensor(buf183, (768, 256), (256, 1), 0), reinterpret_tensor(buf184, (768, ), (1, ), 0), reinterpret_tensor(buf175, (256, 256), (256, 1), 0), reinterpret_tensor(buf176, (256, ), (1, ), 0), buf171, buf172, reinterpret_tensor(buf167, (768, 256), (256, 1), 0), reinterpret_tensor(buf168, (768, ), (1, ), 0), reinterpret_tensor(buf163, (256, 768), (768, 1), 0), reinterpret_tensor(buf164, (256, ), (1, ), 0), buf159, buf160, reinterpret_tensor(buf155, (768, 256), (256, 1), 0), reinterpret_tensor(buf156, (768, ), (1, ), 0), reinterpret_tensor(buf147, (256, 256), (256, 1), 0), reinterpret_tensor(buf148, (256, ), (1, ), 0), buf143, buf144, reinterpret_tensor(buf139, (768, 256), (256, 1), 0), reinterpret_tensor(buf140, (768, ), (1, ), 0), reinterpret_tensor(buf135, (256, 768), (768, 1), 0), reinterpret_tensor(buf136, (256, ), (1, ), 0), buf131, buf132, reinterpret_tensor(buf127, (768, 256), (256, 1), 0), reinterpret_tensor(buf128, (768, ), (1, ), 0), reinterpret_tensor(buf119, (256, 256), (256, 1), 0), reinterpret_tensor(buf120, (256, ), (1, ), 0), buf115, buf116, reinterpret_tensor(buf111, (768, 256), (256, 1), 0), reinterpret_tensor(buf112, (768, ), (1, ), 0), reinterpret_tensor(buf107, (256, 768), (768, 1), 0), reinterpret_tensor(buf108, (256, ), (1, ), 0), buf103, buf104, reinterpret_tensor(buf98, (256, 128), (128, 1), 0), reinterpret_tensor(buf99, (256, ), (1, ), 0), buf93, buf94, reinterpret_tensor(buf88, (128, 256), (256, 1), 0), reinterpret_tensor(buf89, (128, ), (1, ), 0), buf84, buf85, reinterpret_tensor(buf79, (256, 256), (256, 1), 0), reinterpret_tensor(buf78, (256, ), (1, ), 0), reinterpret_tensor(buf76, (256, 256), (256, 1), 0), reinterpret_tensor(buf77, (256, ), (1, ), 0), reinterpret_tensor(buf72, (256, 256), (256, 1), 0), reinterpret_tensor(buf73, (256, ), (1, ), 0), reinterpret_tensor(buf62, (256, 256), (256, 1), 0), reinterpret_tensor(buf63, (256, ), (1, ), 0), buf59, buf60, reinterpret_tensor(buf53, (128, 256), (256, 1), 0), reinterpret_tensor(buf54, (128, ), (1, ), 0), buf50, buf51, reinterpret_tensor(buf45, (128, 128), (128, 1), 0), reinterpret_tensor(buf44, (128, ), (1, ), 0), reinterpret_tensor(buf42, (128, 128), (128, 1), 0), reinterpret_tensor(buf43, (128, ), (1, ), 0), reinterpret_tensor(buf38, (128, 128), (128, 1), 0), reinterpret_tensor(buf39, (128, ), (1, ), 0), reinterpret_tensor(buf28, (128, 128), (128, 1), 0), reinterpret_tensor(buf29, (128, ), (1, ), 0), buf25, buf26, reinterpret_tensor(buf19, (256, 128), (128, 1), 0), reinterpret_tensor(buf20, (256, ), (1, ), 0), buf16, buf17, buf11, buf12, reinterpret_tensor(buf6, (1000, 128), (128, 1), 0), reinterpret_tensor(buf7, (1000, ), (1, ), 0), reinterpret_tensor(buf3, (1000, 256), (256, 1), 0), reinterpret_tensor(buf4, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_5 = rand_strided((128, 3, 12, 12), (432, 1, 36, 3), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((256, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((8, 3, 240, 240), (172800, 1, 720, 3), device='cpu', dtype=torch.float32)
    add_46 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    mul_82 = rand_strided((8, 401, 128), (51328, 128, 1), device='cpu', dtype=torch.float32)
    view_6 = rand_strided((3208, 128), (128, 1), device='cpu', dtype=torch.float32)
    getitem_2 = rand_strided((8, 4, 401, 32), (153984, 32, 384, 1), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((8, 4, 401, 32), (153984, 32, 384, 1), device='cpu', dtype=torch.float32)
    getitem_4 = rand_strided((8, 4, 401, 32), (153984, 32, 384, 1), device='cpu', dtype=torch.float32)
    getitem_6 = rand_strided((8, 4, 401), (1604, 1, 4), device='cpu', dtype=torch.float32)
    getitem_7 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_8 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_11 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_12 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_10 = rand_strided((3208, 128), (128, 1), device='cpu', dtype=torch.float32)
    mul_84 = rand_strided((8, 401, 128), (51328, 128, 1), device='cpu', dtype=torch.float32)
    view_12 = rand_strided((3208, 128), (128, 1), device='cpu', dtype=torch.float32)
    addmm_2 = rand_strided((3208, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_14 = rand_strided((3208, 384), (384, 1), device='cpu', dtype=torch.float32)
    mul_89 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    view_16 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_18 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_19 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_20 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_22 = rand_strided((8, 4, 197), (788, 1, 4), device='cpu', dtype=torch.float32)
    getitem_23 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_24 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_27 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_28 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_20 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    mul_91 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    view_22 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_6 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_24 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_96 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    view_26 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_34 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_35 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_36 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_38 = rand_strided((8, 4, 197), (788, 1, 4), device='cpu', dtype=torch.float32)
    getitem_39 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_40 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_43 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_44 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_30 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    mul_98 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    view_32 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_34 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_103 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    view_36 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_50 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_51 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_52 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_54 = rand_strided((8, 4, 197), (788, 1, 4), device='cpu', dtype=torch.float32)
    getitem_55 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_56 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_59 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_60 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_40 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    mul_105 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    view_42 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_14 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_44 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_110 = rand_strided((8, 1, 128), (128, 128, 1), device='cpu', dtype=torch.float32)
    view_46 = rand_strided((8, 128), (128, 1), device='cpu', dtype=torch.float32)
    mul_115 = rand_strided((8, 1, 256), (256, 256, 1), device='cpu', dtype=torch.float32)
    view_48 = rand_strided((8, 256), (256, 1), device='cpu', dtype=torch.float32)
    cat_2 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    getitem_69 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_10 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    view_50 = rand_strided((8, 256), (50432, 1), device='cpu', dtype=torch.float32)
    view_53 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    view_66 = rand_strided((8, 256), (256, 1), device='cpu', dtype=torch.float32)
    mul_123 = rand_strided((8, 1, 256), (256, 256, 1), device='cpu', dtype=torch.float32)
    view_68 = rand_strided((8, 256), (256, 1), device='cpu', dtype=torch.float32)
    cat_3 = rand_strided((8, 401, 128), (51328, 128, 1), device='cpu', dtype=torch.float32)
    cat_4 = rand_strided((8, 401, 128), (51328, 128, 1), device='cpu', dtype=torch.float32)
    getitem_73 = rand_strided((8, 401, 1), (401, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_12 = rand_strided((8, 401, 1), (401, 1, 1), device='cpu', dtype=torch.float32)
    view_70 = rand_strided((8, 128), (51328, 1), device='cpu', dtype=torch.float32)
    view_73 = rand_strided((3208, 128), (128, 1), device='cpu', dtype=torch.float32)
    view_86 = rand_strided((8, 128), (128, 1), device='cpu', dtype=torch.float32)
    mul_131 = rand_strided((8, 1, 128), (128, 128, 1), device='cpu', dtype=torch.float32)
    view_88 = rand_strided((8, 128), (128, 1), device='cpu', dtype=torch.float32)
    cat_5 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    getitem_77 = rand_strided((8, 401, 1), (401, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_14 = rand_strided((8, 401, 1), (401, 1, 1), device='cpu', dtype=torch.float32)
    view_90 = rand_strided((3208, 128), (128, 1), device='cpu', dtype=torch.float32)
    getitem_78 = rand_strided((8, 4, 401, 32), (153984, 32, 384, 1), device='cpu', dtype=torch.float32)
    getitem_79 = rand_strided((8, 4, 401, 32), (153984, 32, 384, 1), device='cpu', dtype=torch.float32)
    getitem_80 = rand_strided((8, 4, 401, 32), (153984, 32, 384, 1), device='cpu', dtype=torch.float32)
    getitem_82 = rand_strided((8, 4, 401), (1604, 1, 4), device='cpu', dtype=torch.float32)
    getitem_83 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_84 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_87 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_88 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_94 = rand_strided((3208, 128), (128, 1), device='cpu', dtype=torch.float32)
    mul_138 = rand_strided((8, 401, 128), (51328, 128, 1), device='cpu', dtype=torch.float32)
    view_96 = rand_strided((3208, 128), (128, 1), device='cpu', dtype=torch.float32)
    addmm_28 = rand_strided((3208, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_98 = rand_strided((3208, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_93 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_16 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    view_100 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_94 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_95 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_96 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_98 = rand_strided((8, 4, 197), (788, 1, 4), device='cpu', dtype=torch.float32)
    getitem_99 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_100 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_103 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_104 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_104 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    mul_145 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    view_106 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_32 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_108 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_150 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    view_110 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_110 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_111 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_112 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_114 = rand_strided((8, 4, 197), (788, 1, 4), device='cpu', dtype=torch.float32)
    getitem_115 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_116 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_119 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_120 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_114 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    mul_152 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    view_116 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_36 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_118 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_157 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    view_120 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_126 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_127 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_128 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_130 = rand_strided((8, 4, 197), (788, 1, 4), device='cpu', dtype=torch.float32)
    getitem_131 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_132 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_135 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_136 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_124 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    mul_159 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    view_126 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_40 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_128 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_164 = rand_strided((8, 1, 128), (128, 128, 1), device='cpu', dtype=torch.float32)
    view_130 = rand_strided((8, 128), (128, 1), device='cpu', dtype=torch.float32)
    mul_169 = rand_strided((8, 1, 256), (256, 256, 1), device='cpu', dtype=torch.float32)
    view_132 = rand_strided((8, 256), (256, 1), device='cpu', dtype=torch.float32)
    cat_6 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    getitem_145 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_24 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    view_134 = rand_strided((8, 256), (50432, 1), device='cpu', dtype=torch.float32)
    view_137 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    view_150 = rand_strided((8, 256), (256, 1), device='cpu', dtype=torch.float32)
    mul_177 = rand_strided((8, 1, 256), (256, 256, 1), device='cpu', dtype=torch.float32)
    view_152 = rand_strided((8, 256), (256, 1), device='cpu', dtype=torch.float32)
    cat_7 = rand_strided((8, 401, 128), (51328, 128, 1), device='cpu', dtype=torch.float32)
    cat_8 = rand_strided((8, 401, 128), (51328, 128, 1), device='cpu', dtype=torch.float32)
    getitem_149 = rand_strided((8, 401, 1), (401, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_26 = rand_strided((8, 401, 1), (401, 1, 1), device='cpu', dtype=torch.float32)
    view_154 = rand_strided((8, 128), (51328, 1), device='cpu', dtype=torch.float32)
    view_157 = rand_strided((3208, 128), (128, 1), device='cpu', dtype=torch.float32)
    view_170 = rand_strided((8, 128), (128, 1), device='cpu', dtype=torch.float32)
    mul_185 = rand_strided((8, 1, 128), (128, 128, 1), device='cpu', dtype=torch.float32)
    view_172 = rand_strided((8, 128), (128, 1), device='cpu', dtype=torch.float32)
    cat_9 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    getitem_153 = rand_strided((8, 401, 1), (401, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_28 = rand_strided((8, 401, 1), (401, 1, 1), device='cpu', dtype=torch.float32)
    view_174 = rand_strided((3208, 128), (128, 1), device='cpu', dtype=torch.float32)
    getitem_154 = rand_strided((8, 4, 401, 32), (153984, 32, 384, 1), device='cpu', dtype=torch.float32)
    getitem_155 = rand_strided((8, 4, 401, 32), (153984, 32, 384, 1), device='cpu', dtype=torch.float32)
    getitem_156 = rand_strided((8, 4, 401, 32), (153984, 32, 384, 1), device='cpu', dtype=torch.float32)
    getitem_158 = rand_strided((8, 4, 401), (1604, 1, 4), device='cpu', dtype=torch.float32)
    getitem_159 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_160 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_163 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_164 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_178 = rand_strided((3208, 128), (128, 1), device='cpu', dtype=torch.float32)
    mul_192 = rand_strided((8, 401, 128), (51328, 128, 1), device='cpu', dtype=torch.float32)
    view_180 = rand_strided((3208, 128), (128, 1), device='cpu', dtype=torch.float32)
    addmm_54 = rand_strided((3208, 384), (384, 1), device='cpu', dtype=torch.float32)
    view_182 = rand_strided((3208, 384), (384, 1), device='cpu', dtype=torch.float32)
    getitem_169 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_30 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    view_184 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_170 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_171 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_172 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_174 = rand_strided((8, 4, 197), (788, 1, 4), device='cpu', dtype=torch.float32)
    getitem_175 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_176 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_179 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_180 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_188 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    mul_199 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    view_190 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_58 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_192 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_204 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    view_194 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_186 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_187 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_188 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_190 = rand_strided((8, 4, 197), (788, 1, 4), device='cpu', dtype=torch.float32)
    getitem_191 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_192 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_195 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_196 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_198 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    mul_206 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    view_200 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_62 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_202 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_211 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    view_204 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    getitem_202 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_203 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_204 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cpu', dtype=torch.float32)
    getitem_206 = rand_strided((8, 4, 197), (788, 1, 4), device='cpu', dtype=torch.float32)
    getitem_207 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_208 = rand_strided((), (), device='cpu', dtype=torch.int32)
    getitem_211 = rand_strided((), (), device='cpu', dtype=torch.int64)
    getitem_212 = rand_strided((), (), device='cpu', dtype=torch.int64)
    view_208 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    mul_213 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    view_210 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    addmm_66 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_212 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_218 = rand_strided((8, 1, 128), (128, 128, 1), device='cpu', dtype=torch.float32)
    view_214 = rand_strided((8, 128), (128, 1), device='cpu', dtype=torch.float32)
    mul_223 = rand_strided((8, 1, 256), (256, 256, 1), device='cpu', dtype=torch.float32)
    view_216 = rand_strided((8, 256), (256, 1), device='cpu', dtype=torch.float32)
    cat_10 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    getitem_221 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_38 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    view_218 = rand_strided((8, 256), (50432, 1), device='cpu', dtype=torch.float32)
    view_221 = rand_strided((1576, 256), (256, 1), device='cpu', dtype=torch.float32)
    view_234 = rand_strided((8, 256), (256, 1), device='cpu', dtype=torch.float32)
    mul_231 = rand_strided((8, 1, 256), (256, 256, 1), device='cpu', dtype=torch.float32)
    view_236 = rand_strided((8, 256), (256, 1), device='cpu', dtype=torch.float32)
    cat_11 = rand_strided((8, 401, 128), (51328, 128, 1), device='cpu', dtype=torch.float32)
    cat_12 = rand_strided((8, 401, 128), (51328, 128, 1), device='cpu', dtype=torch.float32)
    getitem_225 = rand_strided((8, 401, 1), (401, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_40 = rand_strided((8, 401, 1), (401, 1, 1), device='cpu', dtype=torch.float32)
    view_238 = rand_strided((8, 128), (51328, 1), device='cpu', dtype=torch.float32)
    view_241 = rand_strided((3208, 128), (128, 1), device='cpu', dtype=torch.float32)
    view_254 = rand_strided((8, 128), (128, 1), device='cpu', dtype=torch.float32)
    mul_239 = rand_strided((8, 1, 128), (128, 128, 1), device='cpu', dtype=torch.float32)
    view_256 = rand_strided((8, 128), (128, 1), device='cpu', dtype=torch.float32)
    cat_13 = rand_strided((8, 197, 256), (50432, 256, 1), device='cpu', dtype=torch.float32)
    getitem_229 = rand_strided((8, 401, 1), (401, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_42 = rand_strided((8, 401, 1), (401, 1, 1), device='cpu', dtype=torch.float32)
    getitem_231 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_43 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    clone_68 = rand_strided((8, 128), (128, 1), device='cpu', dtype=torch.float32)
    clone_69 = rand_strided((8, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_142 = rand_strided((1000, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_146 = rand_strided((1000, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_150 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    div_9 = rand_strided((8, 1, 1), (1, 1, 1), device='cpu', dtype=torch.float32)
    permute_154 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_159 = rand_strided((32, 401, 1), (401, 1, 0), device='cpu', dtype=torch.float32)
    permute_160 = rand_strided((32, 32, 401), (12832, 1, 32), device='cpu', dtype=torch.float32)
    alias_18 = rand_strided((8, 4, 1, 401), (1604, 1, 1604, 4), device='cpu', dtype=torch.float32)
    permute_161 = rand_strided((32, 32, 1), (32, 1, 0), device='cpu', dtype=torch.float32)
    permute_162 = rand_strided((32, 401, 32), (12832, 1, 401), device='cpu', dtype=torch.float32)
    permute_165 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_170 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_177 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_179 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_11 = rand_strided((8, 1, 1), (1, 1, 1), device='cpu', dtype=torch.float32)
    permute_183 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_188 = rand_strided((32, 197, 1), (197, 1, 0), device='cpu', dtype=torch.float32)
    permute_189 = rand_strided((32, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    alias_19 = rand_strided((8, 4, 1, 197), (788, 1, 788, 4), device='cpu', dtype=torch.float32)
    permute_190 = rand_strided((32, 64, 1), (64, 1, 0), device='cpu', dtype=torch.float32)
    permute_191 = rand_strided((32, 197, 64), (12608, 1, 197), device='cpu', dtype=torch.float32)
    permute_194 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_199 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_206 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_208 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_13 = rand_strided((8, 1, 1), (1, 1, 1), device='cpu', dtype=torch.float32)
    permute_212 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    div_14 = rand_strided((8, 1, 1), (1, 1, 1), device='cpu', dtype=torch.float32)
    permute_216 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_220 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_15 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_224 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    alias_20 = rand_strided((8, 4, 197, 64), (50432, 1, 256, 4), device='cpu', dtype=torch.float32)
    permute_230 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_16 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_234 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_238 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_17 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_242 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    alias_21 = rand_strided((8, 4, 197, 64), (50432, 1, 256, 4), device='cpu', dtype=torch.float32)
    permute_248 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_18 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_252 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_256 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_19 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_260 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    alias_22 = rand_strided((8, 4, 197, 64), (50432, 1, 256, 4), device='cpu', dtype=torch.float32)
    permute_266 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_270 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_274 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    div_21 = rand_strided((8, 401, 1), (401, 1, 1), device='cpu', dtype=torch.float32)
    permute_278 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    alias_23 = rand_strided((8, 4, 401, 32), (51328, 1, 128, 4), device='cpu', dtype=torch.float32)
    permute_284 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_288 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    div_23 = rand_strided((8, 1, 1), (1, 1, 1), device='cpu', dtype=torch.float32)
    permute_292 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_297 = rand_strided((32, 401, 1), (401, 1, 0), device='cpu', dtype=torch.float32)
    permute_298 = rand_strided((32, 32, 401), (12832, 1, 32), device='cpu', dtype=torch.float32)
    alias_24 = rand_strided((8, 4, 1, 401), (1604, 1, 1604, 4), device='cpu', dtype=torch.float32)
    permute_299 = rand_strided((32, 32, 1), (32, 1, 0), device='cpu', dtype=torch.float32)
    permute_300 = rand_strided((32, 401, 32), (12832, 1, 401), device='cpu', dtype=torch.float32)
    permute_303 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_308 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_315 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_317 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_25 = rand_strided((8, 1, 1), (1, 1, 1), device='cpu', dtype=torch.float32)
    permute_321 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_326 = rand_strided((32, 197, 1), (197, 1, 0), device='cpu', dtype=torch.float32)
    permute_327 = rand_strided((32, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    alias_25 = rand_strided((8, 4, 1, 197), (788, 1, 788, 4), device='cpu', dtype=torch.float32)
    permute_328 = rand_strided((32, 64, 1), (64, 1, 0), device='cpu', dtype=torch.float32)
    permute_329 = rand_strided((32, 197, 64), (12608, 1, 197), device='cpu', dtype=torch.float32)
    permute_332 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_337 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_344 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_346 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_27 = rand_strided((8, 1, 1), (1, 1, 1), device='cpu', dtype=torch.float32)
    permute_350 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    div_28 = rand_strided((8, 1, 1), (1, 1, 1), device='cpu', dtype=torch.float32)
    permute_354 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_358 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_29 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_362 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    alias_26 = rand_strided((8, 4, 197, 64), (50432, 1, 256, 4), device='cpu', dtype=torch.float32)
    permute_368 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_30 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_372 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_376 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_31 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_380 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    alias_27 = rand_strided((8, 4, 197, 64), (50432, 1, 256, 4), device='cpu', dtype=torch.float32)
    permute_386 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_32 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_390 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_394 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_33 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_398 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    alias_28 = rand_strided((8, 4, 197, 64), (50432, 1, 256, 4), device='cpu', dtype=torch.float32)
    permute_404 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_408 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_412 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    div_35 = rand_strided((8, 401, 1), (401, 1, 1), device='cpu', dtype=torch.float32)
    permute_416 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    alias_29 = rand_strided((8, 4, 401, 32), (51328, 1, 128, 4), device='cpu', dtype=torch.float32)
    permute_422 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_426 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    div_37 = rand_strided((8, 1, 1), (1, 1, 1), device='cpu', dtype=torch.float32)
    permute_430 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_435 = rand_strided((32, 401, 1), (401, 1, 0), device='cpu', dtype=torch.float32)
    permute_436 = rand_strided((32, 32, 401), (12832, 1, 32), device='cpu', dtype=torch.float32)
    alias_30 = rand_strided((8, 4, 1, 401), (1604, 1, 1604, 4), device='cpu', dtype=torch.float32)
    permute_437 = rand_strided((32, 32, 1), (32, 1, 0), device='cpu', dtype=torch.float32)
    permute_438 = rand_strided((32, 401, 32), (12832, 1, 401), device='cpu', dtype=torch.float32)
    permute_441 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_446 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_453 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    permute_455 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_39 = rand_strided((8, 1, 1), (1, 1, 1), device='cpu', dtype=torch.float32)
    permute_459 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_464 = rand_strided((32, 197, 1), (197, 1, 0), device='cpu', dtype=torch.float32)
    permute_465 = rand_strided((32, 64, 197), (12608, 1, 64), device='cpu', dtype=torch.float32)
    alias_31 = rand_strided((8, 4, 1, 197), (788, 1, 788, 4), device='cpu', dtype=torch.float32)
    permute_466 = rand_strided((32, 64, 1), (64, 1, 0), device='cpu', dtype=torch.float32)
    permute_467 = rand_strided((32, 197, 64), (12608, 1, 197), device='cpu', dtype=torch.float32)
    permute_470 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_475 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_482 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    permute_484 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_41 = rand_strided((8, 1, 1), (1, 1, 1), device='cpu', dtype=torch.float32)
    permute_488 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    div_42 = rand_strided((8, 1, 1), (1, 1, 1), device='cpu', dtype=torch.float32)
    permute_492 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_496 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_43 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_500 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    alias_32 = rand_strided((8, 4, 197, 64), (50432, 1, 256, 4), device='cpu', dtype=torch.float32)
    permute_506 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_44 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_510 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_514 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_45 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_518 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    alias_33 = rand_strided((8, 4, 197, 64), (50432, 1, 256, 4), device='cpu', dtype=torch.float32)
    permute_524 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_46 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_528 = rand_strided((256, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_532 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_47 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_536 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    alias_34 = rand_strided((8, 4, 197, 64), (50432, 1, 256, 4), device='cpu', dtype=torch.float32)
    permute_542 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    div_48 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_546 = rand_strided((128, 384), (384, 1), device='cpu', dtype=torch.float32)
    permute_550 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    div_49 = rand_strided((8, 401, 1), (401, 1, 1), device='cpu', dtype=torch.float32)
    permute_554 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    alias_35 = rand_strided((8, 4, 401, 32), (51328, 1, 128, 4), device='cpu', dtype=torch.float32)
    permute_560 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    div_50 = rand_strided((8, 401, 1), (401, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_5, primals_7, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_58, primals_61, primals_62, primals_65, primals_75, primals_76, primals_79, primals_89, primals_90, primals_93, primals_99, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_141, primals_142, primals_145, primals_146, primals_149, primals_159, primals_160, primals_163, primals_173, primals_174, primals_177, primals_183, primals_189, primals_195, primals_201, primals_207, primals_213, primals_219, primals_225, primals_226, primals_229, primals_230, primals_233, primals_243, primals_244, primals_247, primals_257, primals_258, primals_261, primals_263, primals_269, add_46, mul_82, view_6, getitem_2, getitem_3, getitem_4, getitem_6, getitem_7, getitem_8, getitem_11, getitem_12, view_10, mul_84, view_12, addmm_2, view_14, mul_89, view_16, getitem_18, getitem_19, getitem_20, getitem_22, getitem_23, getitem_24, getitem_27, getitem_28, view_20, mul_91, view_22, addmm_6, view_24, mul_96, view_26, getitem_34, getitem_35, getitem_36, getitem_38, getitem_39, getitem_40, getitem_43, getitem_44, view_30, mul_98, view_32, addmm_10, view_34, mul_103, view_36, getitem_50, getitem_51, getitem_52, getitem_54, getitem_55, getitem_56, getitem_59, getitem_60, view_40, mul_105, view_42, addmm_14, view_44, mul_110, view_46, mul_115, view_48, cat_2, getitem_69, rsqrt_10, view_50, view_53, view_66, mul_123, view_68, cat_3, cat_4, getitem_73, rsqrt_12, view_70, view_73, view_86, mul_131, view_88, cat_5, getitem_77, rsqrt_14, view_90, getitem_78, getitem_79, getitem_80, getitem_82, getitem_83, getitem_84, getitem_87, getitem_88, view_94, mul_138, view_96, addmm_28, view_98, getitem_93, rsqrt_16, view_100, getitem_94, getitem_95, getitem_96, getitem_98, getitem_99, getitem_100, getitem_103, getitem_104, view_104, mul_145, view_106, addmm_32, view_108, mul_150, view_110, getitem_110, getitem_111, getitem_112, getitem_114, getitem_115, getitem_116, getitem_119, getitem_120, view_114, mul_152, view_116, addmm_36, view_118, mul_157, view_120, getitem_126, getitem_127, getitem_128, getitem_130, getitem_131, getitem_132, getitem_135, getitem_136, view_124, mul_159, view_126, addmm_40, view_128, mul_164, view_130, mul_169, view_132, cat_6, getitem_145, rsqrt_24, view_134, view_137, view_150, mul_177, view_152, cat_7, cat_8, getitem_149, rsqrt_26, view_154, view_157, view_170, mul_185, view_172, cat_9, getitem_153, rsqrt_28, view_174, getitem_154, getitem_155, getitem_156, getitem_158, getitem_159, getitem_160, getitem_163, getitem_164, view_178, mul_192, view_180, addmm_54, view_182, getitem_169, rsqrt_30, view_184, getitem_170, getitem_171, getitem_172, getitem_174, getitem_175, getitem_176, getitem_179, getitem_180, view_188, mul_199, view_190, addmm_58, view_192, mul_204, view_194, getitem_186, getitem_187, getitem_188, getitem_190, getitem_191, getitem_192, getitem_195, getitem_196, view_198, mul_206, view_200, addmm_62, view_202, mul_211, view_204, getitem_202, getitem_203, getitem_204, getitem_206, getitem_207, getitem_208, getitem_211, getitem_212, view_208, mul_213, view_210, addmm_66, view_212, mul_218, view_214, mul_223, view_216, cat_10, getitem_221, rsqrt_38, view_218, view_221, view_234, mul_231, view_236, cat_11, cat_12, getitem_225, rsqrt_40, view_238, view_241, view_254, mul_239, view_256, cat_13, getitem_229, rsqrt_42, getitem_231, rsqrt_43, clone_68, clone_69, permute_142, permute_146, permute_150, div_9, permute_154, permute_159, permute_160, alias_18, permute_161, permute_162, permute_165, permute_170, permute_177, permute_179, div_11, permute_183, permute_188, permute_189, alias_19, permute_190, permute_191, permute_194, permute_199, permute_206, permute_208, div_13, permute_212, div_14, permute_216, permute_220, div_15, permute_224, alias_20, permute_230, div_16, permute_234, permute_238, div_17, permute_242, alias_21, permute_248, div_18, permute_252, permute_256, div_19, permute_260, alias_22, permute_266, permute_270, permute_274, div_21, permute_278, alias_23, permute_284, permute_288, div_23, permute_292, permute_297, permute_298, alias_24, permute_299, permute_300, permute_303, permute_308, permute_315, permute_317, div_25, permute_321, permute_326, permute_327, alias_25, permute_328, permute_329, permute_332, permute_337, permute_344, permute_346, div_27, permute_350, div_28, permute_354, permute_358, div_29, permute_362, alias_26, permute_368, div_30, permute_372, permute_376, div_31, permute_380, alias_27, permute_386, div_32, permute_390, permute_394, div_33, permute_398, alias_28, permute_404, permute_408, permute_412, div_35, permute_416, alias_29, permute_422, permute_426, div_37, permute_430, permute_435, permute_436, alias_30, permute_437, permute_438, permute_441, permute_446, permute_453, permute_455, div_39, permute_459, permute_464, permute_465, alias_31, permute_466, permute_467, permute_470, permute_475, permute_482, permute_484, div_41, permute_488, div_42, permute_492, permute_496, div_43, permute_500, alias_32, permute_506, div_44, permute_510, permute_514, div_45, permute_518, alias_33, permute_524, div_46, permute_528, permute_532, div_47, permute_536, alias_34, permute_542, div_48, permute_546, permute_550, div_49, permute_554, alias_35, permute_560, div_50, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('crossvit_9_240', benchmark_compiled_module)
