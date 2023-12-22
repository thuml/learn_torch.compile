
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9456L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
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
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (620944L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3152L*x2) + (620944L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (620944L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (16L*x3) + (3152L*x2) + (620944L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (3152L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (620944L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3152L*x2) + (620944L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (3152L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.14433756729740643);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (620944L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (620944L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (16L*x3) + (3152L*x2) + (620944L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (3152L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            auto tmp6 = static_cast<float>(0.14433756729740643);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (620944L*x0))] = tmp7;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_5 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(48L); x4+=static_cast<long>(8L))
                            {
                                float tmp13[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>((-1210368L) + x3 + (197L*x4) + (9456L*x2) + (151296L*x1) + (1210368L*x0)), static_cast<long>(197L), tmp13, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                                    auto tmp1 = static_cast<int>(0);
                                    auto tmp2 = tmp0 >= tmp1;
                                    auto tmp3 = static_cast<int>(8);
                                    auto tmp4 = tmp0 < tmp3;
                                    auto tmp5 = [&]
                                    {
                                        auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x4 + (48L*x3) + (48L*x3_inner) + (9456L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
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
                                        auto tmp14 = at::vec::Vectorized<float>::loadu(tmp13 + static_cast<long>(8L*x3_inner));
                                        return tmp14;
                                    }
                                    ;
                                    auto tmp15 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                                    auto tmp16 = tmp0 >= tmp9;
                                    auto tmp17 = static_cast<int>(24);
                                    auto tmp18 = tmp0 < tmp17;
                                    auto tmp19 = [&]
                                    {
                                        auto tmp20 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x4 + (48L*x3) + (48L*x3_inner) + (9456L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp16));
                                        return tmp20;
                                    }
                                    ;
                                    auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp16));
                                    auto tmp22 = to_float_mask(tmp11);
                                    auto tmp23 = decltype(tmp15)::blendv(tmp21, tmp15, tmp22);
                                    auto tmp24 = to_float_mask(tmp4);
                                    auto tmp25 = decltype(tmp7)::blendv(tmp23, tmp7, tmp24);
                                    tmp25.store(out_ptr0 + static_cast<long>(x4 + (48L*x2) + (768L*x0) + (2304L*x3) + (2304L*x3_inner) + (453888L*x1)));
                                }
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(48L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = c10::convert<long>(x1 + (8L*x0));
                                auto tmp1 = static_cast<long>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<long>(8);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = [&]
                                {
                                    auto tmp6 = in_ptr0[static_cast<long>(x4 + (48L*x3) + (9456L*x2) + (151296L*x1) + (1210368L*x0))];
                                    return tmp6;
                                }
                                ;
                                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                                auto tmp8 = tmp0 >= tmp3;
                                auto tmp9 = static_cast<long>(16);
                                auto tmp10 = tmp0 < tmp9;
                                auto tmp11 = tmp8 & tmp10;
                                auto tmp12 = [&]
                                {
                                    auto tmp13 = in_ptr1[static_cast<long>((-1210368L) + x3 + (197L*x4) + (9456L*x2) + (151296L*x1) + (1210368L*x0))];
                                    return tmp13;
                                }
                                ;
                                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                auto tmp15 = tmp0 >= tmp9;
                                auto tmp16 = static_cast<long>(24);
                                auto tmp17 = tmp0 < tmp16;
                                auto tmp18 = [&]
                                {
                                    auto tmp19 = in_ptr2[static_cast<long>((-2420736L) + x4 + (48L*x3) + (9456L*x2) + (151296L*x1) + (1210368L*x0))];
                                    return tmp19;
                                }
                                ;
                                auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                                auto tmp21 = tmp11 ? tmp14 : tmp20;
                                auto tmp22 = tmp4 ? tmp7 : tmp21;
                                out_ptr0[static_cast<long>(x4 + (48L*x2) + (768L*x0) + (2304L*x3) + (453888L*x1))] = tmp22;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
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
                    auto tmp1 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
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


cpp_fused_gelu_gelu_backward_sum_7 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9456L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_mul_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
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
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (620944L*x0)));
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3152L*x2) + (620944L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 * tmp1;
                                tmp_acc0_vec = tmp_acc0_vec + tmp2;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (620944L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x1 + (16L*x3) + (3152L*x2) + (620944L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                tmp_acc0 = tmp_acc0 + tmp2;
                            }
                            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                            out_ptr0[static_cast<long>(x2 + (197L*x1) + (3152L*x0))] = static_cast<float>(tmp_acc0);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (620944L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr1[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3152L*x2) + (620944L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (3152L*x0))];
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp1 * tmp4;
                            auto tmp6 = tmp2 - tmp5;
                            auto tmp7 = static_cast<float>(0.14433756729740643);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (620944L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (620944L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (16L*x3) + (3152L*x2) + (620944L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (197L*x1) + (3152L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
                            auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                            auto tmp6 = static_cast<float>(0.14433756729740643);
                            auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                            in_out_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (620944L*x0))] = tmp7;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_11 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(48L); x4+=static_cast<long>(8L))
                            {
                                float tmp13[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>((-1210368L) + x3 + (197L*x4) + (9456L*x2) + (151296L*x1) + (1210368L*x0)), static_cast<long>(197L), tmp13, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp0 = c10::convert<int>(x1 + (8L*x0));
                                    auto tmp1 = static_cast<int>(0);
                                    auto tmp2 = tmp0 >= tmp1;
                                    auto tmp3 = static_cast<int>(8);
                                    auto tmp4 = tmp0 < tmp3;
                                    auto tmp5 = [&]
                                    {
                                        auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x4 + (48L*x3) + (48L*x3_inner) + (9456L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp4));
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
                                        auto tmp14 = at::vec::Vectorized<float>::loadu(tmp13 + static_cast<long>(8L*x3_inner));
                                        return tmp14;
                                    }
                                    ;
                                    auto tmp15 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                                    auto tmp16 = tmp0 >= tmp9;
                                    auto tmp17 = static_cast<int>(24);
                                    auto tmp18 = tmp0 < tmp17;
                                    auto tmp19 = [&]
                                    {
                                        auto tmp20 = masked_load(in_ptr2 + static_cast<long>((-2420736L) + x4 + (48L*x3) + (48L*x3_inner) + (9456L*x2) + (151296L*x1) + (1210368L*x0)), to_float_mask(tmp16));
                                        return tmp20;
                                    }
                                    ;
                                    auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp16));
                                    auto tmp22 = to_float_mask(tmp11);
                                    auto tmp23 = decltype(tmp15)::blendv(tmp21, tmp15, tmp22);
                                    auto tmp24 = to_float_mask(tmp4);
                                    auto tmp25 = decltype(tmp7)::blendv(tmp23, tmp7, tmp24);
                                    tmp25.store(out_ptr0 + static_cast<long>(x4 + (48L*x2) + (768L*x0) + (2304L*x3) + (2304L*x3_inner) + (453888L*x1)));
                                }
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(48L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = c10::convert<long>(x1 + (8L*x0));
                                auto tmp1 = static_cast<long>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<long>(8);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = [&]
                                {
                                    auto tmp6 = in_ptr0[static_cast<long>(x4 + (48L*x3) + (9456L*x2) + (151296L*x1) + (1210368L*x0))];
                                    return tmp6;
                                }
                                ;
                                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                                auto tmp8 = tmp0 >= tmp3;
                                auto tmp9 = static_cast<long>(16);
                                auto tmp10 = tmp0 < tmp9;
                                auto tmp11 = tmp8 & tmp10;
                                auto tmp12 = [&]
                                {
                                    auto tmp13 = in_ptr1[static_cast<long>((-1210368L) + x3 + (197L*x4) + (9456L*x2) + (151296L*x1) + (1210368L*x0))];
                                    return tmp13;
                                }
                                ;
                                auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                                auto tmp15 = tmp0 >= tmp9;
                                auto tmp16 = static_cast<long>(24);
                                auto tmp17 = tmp0 < tmp16;
                                auto tmp18 = [&]
                                {
                                    auto tmp19 = in_ptr2[static_cast<long>((-2420736L) + x4 + (48L*x3) + (9456L*x2) + (151296L*x1) + (1210368L*x0))];
                                    return tmp19;
                                }
                                ;
                                auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                                auto tmp21 = tmp11 ? tmp14 : tmp20;
                                auto tmp22 = tmp4 ? tmp7 : tmp21;
                                out_ptr0[static_cast<long>(x4 + (48L*x2) + (768L*x0) + (2304L*x3) + (453888L*x1))] = tmp22;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_add_clone_native_layer_norm_native_layer_norm_backward_12 = async_compile.cpp('''
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = in_ptr3[static_cast<long>(x0)];
                        auto tmp7 = in_ptr4[static_cast<long>(x0)];
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
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = tmp_acc1 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc1_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = in_ptr3[static_cast<long>(x1)];
                        auto tmp5 = in_ptr4[static_cast<long>(x1)];
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 - tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = tmp0 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
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
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp13 = in_ptr3[static_cast<long>(x0)];
                    auto tmp18 = out_ptr1[static_cast<long>(x0)];
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(768L + x1 + (768L*(static_cast<long>(x0) % static_cast<long>(196L))) + (151296L*(c10::div_floor_integer(x0, 196L)))));
                    tmp0.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                }
            }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(768L + x2 + (768L*x1) + (151296L*x0)));
                        auto tmp1 = in_ptr5[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp8 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                        auto tmp12 = out_ptr2[static_cast<long>(x1 + (196L*x0))];
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
                        tmp18.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                                auto tmp1 = tmp0.neg();
                                auto tmp3 = decltype(tmp2)(1)/(decltype(tmp2)(1) + tmp2.neg().exp());
                                auto tmp4 = static_cast<float>(1.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp5 - tmp3;
                                auto tmp8 = tmp6 * tmp7;
                                auto tmp10 = tmp3 * tmp9;
                                auto tmp11 = tmp8 + tmp10;
                                auto tmp13 = tmp11 / tmp12;
                                auto tmp14 = tmp13 / tmp12;
                                auto tmp15 = tmp1 * tmp14;
                                tmp_acc0_vec = tmp_acc0_vec + tmp15;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp4 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x1)];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp17 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp6 * tmp9;
                            auto tmp12 = tmp10 * tmp11;
                            auto tmp13 = static_cast<float>(1.0);
                            auto tmp14 = decltype(tmp13)(tmp13 - tmp8);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp6 * tmp15;
                            auto tmp18 = tmp16 * tmp17;
                            tmp12.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            tmp18.store(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x1)];
                            auto tmp8 = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp13 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp5)(1) / (decltype(tmp5)(1) + std::exp(-tmp5));
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp10)(tmp10 - tmp6);
                            auto tmp12 = decltype(tmp4)(tmp4 * tmp11);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp9;
                            out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (16L*x2) + (3136L*x1)));
                                auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x3) + (3136L*x2) + (614656L*x1)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                auto tmp6 = tmp4 * tmp5;
                                auto tmp8 = tmp4 * tmp7;
                                auto tmp11 = tmp5 * tmp10;
                                auto tmp12 = tmp9 - tmp11;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                tmp_acc1_vec = tmp_acc1_vec + tmp8;
                                tmp_acc2_vec = tmp_acc2_vec + tmp12;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    auto tmp9 = tmp8.neg();
                    auto tmp10 = tmp9 * tmp6;
                    auto tmp11 = tmp7 + tmp10;
                    tmp11.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr7[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            auto tmp6 = static_cast<float>(0.14433756729740643);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp8.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            auto tmp5 = static_cast<float>(0.14433756729740643);
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))] = tmp4;
                        }
                    }
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(48L); x4+=static_cast<long>(1L))
                            {
                                auto tmp3 = in_ptr0[static_cast<long>(x1 + (196L*x4) + (9408L*x3) + (150528L*x0))];
                                auto tmp8 = in_ptr1[static_cast<long>(x4 + (48L*x1) + (9408L*x3) + (150528L*x0))];
                                auto tmp0 = c10::convert<int>(x2);
                                auto tmp1 = static_cast<int>(1);
                                auto tmp2 = tmp0 == tmp1;
                                auto tmp4 = static_cast<float>(0.0);
                                auto tmp5 = tmp2 ? tmp3 : tmp4;
                                auto tmp6 = static_cast<int>(0);
                                auto tmp7 = tmp0 == tmp6;
                                auto tmp9 = tmp7 ? tmp8 : tmp4;
                                auto tmp10 = decltype(tmp5)(tmp5 + tmp9);
                                out_ptr0[static_cast<long>(x4 + (48L*x3) + (768L*x2) + (1536L*x1) + (301056L*x0))] = tmp10;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_20 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp14 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 - tmp11;
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp12 - tmp16;
                    auto tmp18 = at::vec::Vectorized<float>(tmp1);
                    auto tmp19 = tmp18 * tmp17;
                    auto tmp20 = tmp0 + tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_sum_23 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                                auto tmp1 = tmp0.neg();
                                auto tmp3 = decltype(tmp2)(1)/(decltype(tmp2)(1) + tmp2.neg().exp());
                                auto tmp4 = static_cast<float>(1.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp5 - tmp3;
                                auto tmp8 = tmp6 * tmp7;
                                auto tmp10 = tmp3 * tmp9;
                                auto tmp11 = tmp8 + tmp10;
                                auto tmp13 = tmp11 / tmp12;
                                auto tmp14 = tmp13 / tmp12;
                                auto tmp15 = tmp1 * tmp14;
                                tmp_acc0_vec = tmp_acc0_vec + tmp15;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp4 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x1)];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp17 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp6 * tmp9;
                            auto tmp12 = tmp10 * tmp11;
                            auto tmp13 = static_cast<float>(1.0);
                            auto tmp14 = decltype(tmp13)(tmp13 - tmp8);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp6 * tmp15;
                            auto tmp18 = tmp16 * tmp17;
                            tmp12.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            tmp18.store(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x1)];
                            auto tmp8 = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp13 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp5)(1) / (decltype(tmp5)(1) + std::exp(-tmp5));
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp10)(tmp10 - tmp6);
                            auto tmp12 = decltype(tmp4)(tmp4 * tmp11);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp9;
                            out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (16L*x2) + (3136L*x1)));
                                auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x3) + (3136L*x2) + (614656L*x1)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                auto tmp6 = tmp4 * tmp5;
                                auto tmp8 = tmp4 * tmp7;
                                auto tmp11 = tmp5 * tmp10;
                                auto tmp12 = tmp9 - tmp11;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                tmp_acc1_vec = tmp_acc1_vec + tmp8;
                                tmp_acc2_vec = tmp_acc2_vec + tmp12;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    auto tmp9 = tmp8.neg();
                    auto tmp10 = tmp9 * tmp6;
                    auto tmp11 = tmp7 + tmp10;
                    tmp11.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr7[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            auto tmp6 = static_cast<float>(0.14433756729740643);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp8.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            auto tmp5 = static_cast<float>(0.14433756729740643);
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp6;
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))] = tmp4;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(48L); x4+=static_cast<long>(1L))
                            {
                                auto tmp3 = in_ptr0[static_cast<long>(x1 + (196L*x4) + (9408L*x3) + (150528L*x0))];
                                auto tmp8 = in_ptr1[static_cast<long>(x4 + (48L*x1) + (9408L*x3) + (150528L*x0))];
                                auto tmp0 = c10::convert<int>(x2);
                                auto tmp1 = static_cast<int>(1);
                                auto tmp2 = tmp0 == tmp1;
                                auto tmp4 = static_cast<float>(0.0);
                                auto tmp5 = tmp2 ? tmp3 : tmp4;
                                auto tmp6 = static_cast<int>(0);
                                auto tmp7 = tmp0 == tmp6;
                                auto tmp9 = tmp7 ? tmp8 : tmp4;
                                auto tmp10 = decltype(tmp5)(tmp5 + tmp9);
                                out_ptr0[static_cast<long>(x4 + (48L*x3) + (768L*x2) + (1536L*x1) + (301056L*x0))] = tmp10;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_28 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp14 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 - tmp11;
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp12 - tmp16;
                    auto tmp18 = at::vec::Vectorized<float>(tmp1);
                    auto tmp19 = tmp18 * tmp17;
                    auto tmp20 = tmp0 + tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                                auto tmp1 = tmp0.neg();
                                auto tmp3 = decltype(tmp2)(1)/(decltype(tmp2)(1) + tmp2.neg().exp());
                                auto tmp4 = static_cast<float>(1.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp5 - tmp3;
                                auto tmp8 = tmp6 * tmp7;
                                auto tmp10 = tmp3 * tmp9;
                                auto tmp11 = tmp8 + tmp10;
                                auto tmp13 = tmp11 / tmp12;
                                auto tmp14 = tmp13 / tmp12;
                                auto tmp15 = tmp1 * tmp14;
                                tmp_acc0_vec = tmp_acc0_vec + tmp15;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp4 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x1)];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp17 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp6 * tmp9;
                            auto tmp12 = tmp10 * tmp11;
                            auto tmp13 = static_cast<float>(1.0);
                            auto tmp14 = decltype(tmp13)(tmp13 - tmp8);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp6 * tmp15;
                            auto tmp18 = tmp16 * tmp17;
                            tmp12.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            tmp18.store(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x1)];
                            auto tmp8 = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp13 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp5)(1) / (decltype(tmp5)(1) + std::exp(-tmp5));
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp10)(tmp10 - tmp6);
                            auto tmp12 = decltype(tmp4)(tmp4 * tmp11);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp9;
                            out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (16L*x2) + (3136L*x1)));
                                auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x3) + (3136L*x2) + (614656L*x1)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                auto tmp6 = tmp4 * tmp5;
                                auto tmp8 = tmp4 * tmp7;
                                auto tmp11 = tmp5 * tmp10;
                                auto tmp12 = tmp9 - tmp11;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                tmp_acc1_vec = tmp_acc1_vec + tmp8;
                                tmp_acc2_vec = tmp_acc2_vec + tmp12;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    auto tmp9 = tmp8.neg();
                    auto tmp10 = tmp9 * tmp6;
                    auto tmp11 = tmp7 + tmp10;
                    tmp11.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr7[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            auto tmp6 = static_cast<float>(0.14433756729740643);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp8.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            auto tmp5 = static_cast<float>(0.14433756729740643);
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp6;
                        }
                    }
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))] = tmp4;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(48L); x4+=static_cast<long>(1L))
                            {
                                auto tmp3 = in_ptr0[static_cast<long>(x1 + (196L*x4) + (9408L*x3) + (150528L*x0))];
                                auto tmp8 = in_ptr1[static_cast<long>(x4 + (48L*x1) + (9408L*x3) + (150528L*x0))];
                                auto tmp0 = c10::convert<int>(x2);
                                auto tmp1 = static_cast<int>(1);
                                auto tmp2 = tmp0 == tmp1;
                                auto tmp4 = static_cast<float>(0.0);
                                auto tmp5 = tmp2 ? tmp3 : tmp4;
                                auto tmp6 = static_cast<int>(0);
                                auto tmp7 = tmp0 == tmp6;
                                auto tmp9 = tmp7 ? tmp8 : tmp4;
                                auto tmp10 = decltype(tmp5)(tmp5 + tmp9);
                                out_ptr0[static_cast<long>(x4 + (48L*x3) + (768L*x2) + (1536L*x1) + (301056L*x0))] = tmp10;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_36 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp14 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 - tmp11;
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp12 - tmp16;
                    auto tmp18 = at::vec::Vectorized<float>(tmp1);
                    auto tmp19 = tmp18 * tmp17;
                    auto tmp20 = tmp0 + tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_37 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                                auto tmp1 = tmp0.neg();
                                auto tmp3 = decltype(tmp2)(1)/(decltype(tmp2)(1) + tmp2.neg().exp());
                                auto tmp4 = static_cast<float>(1.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp5 - tmp3;
                                auto tmp8 = tmp6 * tmp7;
                                auto tmp10 = tmp3 * tmp9;
                                auto tmp11 = tmp8 + tmp10;
                                auto tmp13 = tmp11 / tmp12;
                                auto tmp14 = tmp13 / tmp12;
                                auto tmp15 = tmp1 * tmp14;
                                tmp_acc0_vec = tmp_acc0_vec + tmp15;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp4 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x1)];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp17 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp6 * tmp9;
                            auto tmp12 = tmp10 * tmp11;
                            auto tmp13 = static_cast<float>(1.0);
                            auto tmp14 = decltype(tmp13)(tmp13 - tmp8);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp6 * tmp15;
                            auto tmp18 = tmp16 * tmp17;
                            tmp12.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            tmp18.store(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x1)];
                            auto tmp8 = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp13 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp5)(1) / (decltype(tmp5)(1) + std::exp(-tmp5));
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp10)(tmp10 - tmp6);
                            auto tmp12 = decltype(tmp4)(tmp4 * tmp11);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp9;
                            out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (16L*x2) + (3136L*x1)));
                                auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x3) + (3136L*x2) + (614656L*x1)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                auto tmp6 = tmp4 * tmp5;
                                auto tmp8 = tmp4 * tmp7;
                                auto tmp11 = tmp5 * tmp10;
                                auto tmp12 = tmp9 - tmp11;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                tmp_acc1_vec = tmp_acc1_vec + tmp8;
                                tmp_acc2_vec = tmp_acc2_vec + tmp12;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    auto tmp9 = tmp8.neg();
                    auto tmp10 = tmp9 * tmp6;
                    auto tmp11 = tmp7 + tmp10;
                    tmp11.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr7[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            auto tmp6 = static_cast<float>(0.14433756729740643);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp8.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            auto tmp5 = static_cast<float>(0.14433756729740643);
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))] = tmp4;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(48L); x4+=static_cast<long>(1L))
                            {
                                auto tmp3 = in_ptr0[static_cast<long>(x1 + (196L*x4) + (9408L*x3) + (150528L*x0))];
                                auto tmp8 = in_ptr1[static_cast<long>(x4 + (48L*x1) + (9408L*x3) + (150528L*x0))];
                                auto tmp0 = c10::convert<int>(x2);
                                auto tmp1 = static_cast<int>(1);
                                auto tmp2 = tmp0 == tmp1;
                                auto tmp4 = static_cast<float>(0.0);
                                auto tmp5 = tmp2 ? tmp3 : tmp4;
                                auto tmp6 = static_cast<int>(0);
                                auto tmp7 = tmp0 == tmp6;
                                auto tmp9 = tmp7 ? tmp8 : tmp4;
                                auto tmp10 = decltype(tmp5)(tmp5 + tmp9);
                                out_ptr0[static_cast<long>(x4 + (48L*x3) + (768L*x2) + (1536L*x1) + (301056L*x0))] = tmp10;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_44 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp14 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 - tmp11;
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp12 - tmp16;
                    auto tmp18 = at::vec::Vectorized<float>(tmp1);
                    auto tmp19 = tmp18 * tmp17;
                    auto tmp20 = tmp0 + tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_45 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_46 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_sum_47 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                                auto tmp1 = tmp0.neg();
                                auto tmp3 = decltype(tmp2)(1)/(decltype(tmp2)(1) + tmp2.neg().exp());
                                auto tmp4 = static_cast<float>(1.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp5 - tmp3;
                                auto tmp8 = tmp6 * tmp7;
                                auto tmp10 = tmp3 * tmp9;
                                auto tmp11 = tmp8 + tmp10;
                                auto tmp13 = tmp11 / tmp12;
                                auto tmp14 = tmp13 / tmp12;
                                auto tmp15 = tmp1 * tmp14;
                                tmp_acc0_vec = tmp_acc0_vec + tmp15;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp4 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x1)];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp17 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp6 * tmp9;
                            auto tmp12 = tmp10 * tmp11;
                            auto tmp13 = static_cast<float>(1.0);
                            auto tmp14 = decltype(tmp13)(tmp13 - tmp8);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp6 * tmp15;
                            auto tmp18 = tmp16 * tmp17;
                            tmp12.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            tmp18.store(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x1)];
                            auto tmp8 = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp13 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp5)(1) / (decltype(tmp5)(1) + std::exp(-tmp5));
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp10)(tmp10 - tmp6);
                            auto tmp12 = decltype(tmp4)(tmp4 * tmp11);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp9;
                            out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (16L*x2) + (3136L*x1)));
                                auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x3) + (3136L*x2) + (614656L*x1)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                auto tmp6 = tmp4 * tmp5;
                                auto tmp8 = tmp4 * tmp7;
                                auto tmp11 = tmp5 * tmp10;
                                auto tmp12 = tmp9 - tmp11;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                tmp_acc1_vec = tmp_acc1_vec + tmp8;
                                tmp_acc2_vec = tmp_acc2_vec + tmp12;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    auto tmp9 = tmp8.neg();
                    auto tmp10 = tmp9 * tmp6;
                    auto tmp11 = tmp7 + tmp10;
                    tmp11.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr7[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            auto tmp6 = static_cast<float>(0.14433756729740643);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp8.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            auto tmp5 = static_cast<float>(0.14433756729740643);
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))] = tmp4;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(48L); x4+=static_cast<long>(1L))
                            {
                                auto tmp3 = in_ptr0[static_cast<long>(x1 + (196L*x4) + (9408L*x3) + (150528L*x0))];
                                auto tmp8 = in_ptr1[static_cast<long>(x4 + (48L*x1) + (9408L*x3) + (150528L*x0))];
                                auto tmp0 = c10::convert<int>(x2);
                                auto tmp1 = static_cast<int>(1);
                                auto tmp2 = tmp0 == tmp1;
                                auto tmp4 = static_cast<float>(0.0);
                                auto tmp5 = tmp2 ? tmp3 : tmp4;
                                auto tmp6 = static_cast<int>(0);
                                auto tmp7 = tmp0 == tmp6;
                                auto tmp9 = tmp7 ? tmp8 : tmp4;
                                auto tmp10 = decltype(tmp5)(tmp5 + tmp9);
                                out_ptr0[static_cast<long>(x4 + (48L*x3) + (768L*x2) + (1536L*x1) + (301056L*x0))] = tmp10;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_52 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp14 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 - tmp11;
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp12 - tmp16;
                    auto tmp18 = at::vec::Vectorized<float>(tmp1);
                    auto tmp19 = tmp18 * tmp17;
                    auto tmp20 = tmp0 + tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_53 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_sum_55 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                                auto tmp1 = tmp0.neg();
                                auto tmp3 = decltype(tmp2)(1)/(decltype(tmp2)(1) + tmp2.neg().exp());
                                auto tmp4 = static_cast<float>(1.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp5 - tmp3;
                                auto tmp8 = tmp6 * tmp7;
                                auto tmp10 = tmp3 * tmp9;
                                auto tmp11 = tmp8 + tmp10;
                                auto tmp13 = tmp11 / tmp12;
                                auto tmp14 = tmp13 / tmp12;
                                auto tmp15 = tmp1 * tmp14;
                                tmp_acc0_vec = tmp_acc0_vec + tmp15;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp4 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x1)];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp17 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp6 * tmp9;
                            auto tmp12 = tmp10 * tmp11;
                            auto tmp13 = static_cast<float>(1.0);
                            auto tmp14 = decltype(tmp13)(tmp13 - tmp8);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp6 * tmp15;
                            auto tmp18 = tmp16 * tmp17;
                            tmp12.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            tmp18.store(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x1)];
                            auto tmp8 = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp13 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp5)(1) / (decltype(tmp5)(1) + std::exp(-tmp5));
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp10)(tmp10 - tmp6);
                            auto tmp12 = decltype(tmp4)(tmp4 * tmp11);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp9;
                            out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (16L*x2) + (3136L*x1)));
                                auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x3) + (3136L*x2) + (614656L*x1)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                auto tmp6 = tmp4 * tmp5;
                                auto tmp8 = tmp4 * tmp7;
                                auto tmp11 = tmp5 * tmp10;
                                auto tmp12 = tmp9 - tmp11;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                tmp_acc1_vec = tmp_acc1_vec + tmp8;
                                tmp_acc2_vec = tmp_acc2_vec + tmp12;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    auto tmp9 = tmp8.neg();
                    auto tmp10 = tmp9 * tmp6;
                    auto tmp11 = tmp7 + tmp10;
                    tmp11.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr7[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            auto tmp6 = static_cast<float>(0.14433756729740643);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp8.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            auto tmp5 = static_cast<float>(0.14433756729740643);
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))] = tmp4;
                        }
                    }
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(48L); x4+=static_cast<long>(1L))
                            {
                                auto tmp3 = in_ptr0[static_cast<long>(x1 + (196L*x4) + (9408L*x3) + (150528L*x0))];
                                auto tmp8 = in_ptr1[static_cast<long>(x4 + (48L*x1) + (9408L*x3) + (150528L*x0))];
                                auto tmp0 = c10::convert<int>(x2);
                                auto tmp1 = static_cast<int>(1);
                                auto tmp2 = tmp0 == tmp1;
                                auto tmp4 = static_cast<float>(0.0);
                                auto tmp5 = tmp2 ? tmp3 : tmp4;
                                auto tmp6 = static_cast<int>(0);
                                auto tmp7 = tmp0 == tmp6;
                                auto tmp9 = tmp7 ? tmp8 : tmp4;
                                auto tmp10 = decltype(tmp5)(tmp5 + tmp9);
                                out_ptr0[static_cast<long>(x4 + (48L*x3) + (768L*x2) + (1536L*x1) + (301056L*x0))] = tmp10;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_60 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp14 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 - tmp11;
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp12 - tmp16;
                    auto tmp18 = at::vec::Vectorized<float>(tmp1);
                    auto tmp19 = tmp18 * tmp17;
                    auto tmp20 = tmp0 + tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_61 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                                auto tmp1 = tmp0.neg();
                                auto tmp3 = decltype(tmp2)(1)/(decltype(tmp2)(1) + tmp2.neg().exp());
                                auto tmp4 = static_cast<float>(1.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp5 - tmp3;
                                auto tmp8 = tmp6 * tmp7;
                                auto tmp10 = tmp3 * tmp9;
                                auto tmp11 = tmp8 + tmp10;
                                auto tmp13 = tmp11 / tmp12;
                                auto tmp14 = tmp13 / tmp12;
                                auto tmp15 = tmp1 * tmp14;
                                tmp_acc0_vec = tmp_acc0_vec + tmp15;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp4 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x1)];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp17 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp6 * tmp9;
                            auto tmp12 = tmp10 * tmp11;
                            auto tmp13 = static_cast<float>(1.0);
                            auto tmp14 = decltype(tmp13)(tmp13 - tmp8);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp6 * tmp15;
                            auto tmp18 = tmp16 * tmp17;
                            tmp12.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            tmp18.store(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x1)];
                            auto tmp8 = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp13 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp5)(1) / (decltype(tmp5)(1) + std::exp(-tmp5));
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp10)(tmp10 - tmp6);
                            auto tmp12 = decltype(tmp4)(tmp4 * tmp11);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp9;
                            out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (16L*x2) + (3136L*x1)));
                                auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x3) + (3136L*x2) + (614656L*x1)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                auto tmp6 = tmp4 * tmp5;
                                auto tmp8 = tmp4 * tmp7;
                                auto tmp11 = tmp5 * tmp10;
                                auto tmp12 = tmp9 - tmp11;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                tmp_acc1_vec = tmp_acc1_vec + tmp8;
                                tmp_acc2_vec = tmp_acc2_vec + tmp12;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    auto tmp9 = tmp8.neg();
                    auto tmp10 = tmp9 * tmp6;
                    auto tmp11 = tmp7 + tmp10;
                    tmp11.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr7[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            auto tmp6 = static_cast<float>(0.14433756729740643);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp8.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            auto tmp5 = static_cast<float>(0.14433756729740643);
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp6;
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))] = tmp4;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(48L); x4+=static_cast<long>(1L))
                            {
                                auto tmp3 = in_ptr0[static_cast<long>(x1 + (196L*x4) + (9408L*x3) + (150528L*x0))];
                                auto tmp8 = in_ptr1[static_cast<long>(x4 + (48L*x1) + (9408L*x3) + (150528L*x0))];
                                auto tmp0 = c10::convert<int>(x2);
                                auto tmp1 = static_cast<int>(1);
                                auto tmp2 = tmp0 == tmp1;
                                auto tmp4 = static_cast<float>(0.0);
                                auto tmp5 = tmp2 ? tmp3 : tmp4;
                                auto tmp6 = static_cast<int>(0);
                                auto tmp7 = tmp0 == tmp6;
                                auto tmp9 = tmp7 ? tmp8 : tmp4;
                                auto tmp10 = decltype(tmp5)(tmp5 + tmp9);
                                out_ptr0[static_cast<long>(x4 + (48L*x3) + (768L*x2) + (1536L*x1) + (301056L*x0))] = tmp10;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_68 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp14 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 - tmp11;
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp12 - tmp16;
                    auto tmp18 = at::vec::Vectorized<float>(tmp1);
                    auto tmp19 = tmp18 * tmp17;
                    auto tmp20 = tmp0 + tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_69 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_70 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_sum_71 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                                auto tmp1 = tmp0.neg();
                                auto tmp3 = decltype(tmp2)(1)/(decltype(tmp2)(1) + tmp2.neg().exp());
                                auto tmp4 = static_cast<float>(1.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp5 - tmp3;
                                auto tmp8 = tmp6 * tmp7;
                                auto tmp10 = tmp3 * tmp9;
                                auto tmp11 = tmp8 + tmp10;
                                auto tmp13 = tmp11 / tmp12;
                                auto tmp14 = tmp13 / tmp12;
                                auto tmp15 = tmp1 * tmp14;
                                tmp_acc0_vec = tmp_acc0_vec + tmp15;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp4 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x1)];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp17 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp6 * tmp9;
                            auto tmp12 = tmp10 * tmp11;
                            auto tmp13 = static_cast<float>(1.0);
                            auto tmp14 = decltype(tmp13)(tmp13 - tmp8);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp6 * tmp15;
                            auto tmp18 = tmp16 * tmp17;
                            tmp12.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            tmp18.store(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x1)];
                            auto tmp8 = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp13 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp5)(1) / (decltype(tmp5)(1) + std::exp(-tmp5));
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp10)(tmp10 - tmp6);
                            auto tmp12 = decltype(tmp4)(tmp4 * tmp11);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp9;
                            out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (16L*x2) + (3136L*x1)));
                                auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x3) + (3136L*x2) + (614656L*x1)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                auto tmp6 = tmp4 * tmp5;
                                auto tmp8 = tmp4 * tmp7;
                                auto tmp11 = tmp5 * tmp10;
                                auto tmp12 = tmp9 - tmp11;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                tmp_acc1_vec = tmp_acc1_vec + tmp8;
                                tmp_acc2_vec = tmp_acc2_vec + tmp12;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    auto tmp9 = tmp8.neg();
                    auto tmp10 = tmp9 * tmp6;
                    auto tmp11 = tmp7 + tmp10;
                    tmp11.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr7[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            auto tmp6 = static_cast<float>(0.14433756729740643);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp8.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            auto tmp5 = static_cast<float>(0.14433756729740643);
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))] = tmp4;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(48L); x4+=static_cast<long>(1L))
                            {
                                auto tmp3 = in_ptr0[static_cast<long>(x1 + (196L*x4) + (9408L*x3) + (150528L*x0))];
                                auto tmp8 = in_ptr1[static_cast<long>(x4 + (48L*x1) + (9408L*x3) + (150528L*x0))];
                                auto tmp0 = c10::convert<int>(x2);
                                auto tmp1 = static_cast<int>(1);
                                auto tmp2 = tmp0 == tmp1;
                                auto tmp4 = static_cast<float>(0.0);
                                auto tmp5 = tmp2 ? tmp3 : tmp4;
                                auto tmp6 = static_cast<int>(0);
                                auto tmp7 = tmp0 == tmp6;
                                auto tmp9 = tmp7 ? tmp8 : tmp4;
                                auto tmp10 = decltype(tmp5)(tmp5 + tmp9);
                                out_ptr0[static_cast<long>(x4 + (48L*x3) + (768L*x2) + (1536L*x1) + (301056L*x0))] = tmp10;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_76 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp14 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 - tmp11;
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp12 - tmp16;
                    auto tmp18 = at::vec::Vectorized<float>(tmp1);
                    auto tmp19 = tmp18 * tmp17;
                    auto tmp20 = tmp0 + tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_77 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_78 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_sum_79 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                                auto tmp1 = tmp0.neg();
                                auto tmp3 = decltype(tmp2)(1)/(decltype(tmp2)(1) + tmp2.neg().exp());
                                auto tmp4 = static_cast<float>(1.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp5 - tmp3;
                                auto tmp8 = tmp6 * tmp7;
                                auto tmp10 = tmp3 * tmp9;
                                auto tmp11 = tmp8 + tmp10;
                                auto tmp13 = tmp11 / tmp12;
                                auto tmp14 = tmp13 / tmp12;
                                auto tmp15 = tmp1 * tmp14;
                                tmp_acc0_vec = tmp_acc0_vec + tmp15;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp4 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x1)];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp17 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp6 * tmp9;
                            auto tmp12 = tmp10 * tmp11;
                            auto tmp13 = static_cast<float>(1.0);
                            auto tmp14 = decltype(tmp13)(tmp13 - tmp8);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp6 * tmp15;
                            auto tmp18 = tmp16 * tmp17;
                            tmp12.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            tmp18.store(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x1)];
                            auto tmp8 = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp13 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp5)(1) / (decltype(tmp5)(1) + std::exp(-tmp5));
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp10)(tmp10 - tmp6);
                            auto tmp12 = decltype(tmp4)(tmp4 * tmp11);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp9;
                            out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (16L*x2) + (3136L*x1)));
                                auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x3) + (3136L*x2) + (614656L*x1)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                auto tmp6 = tmp4 * tmp5;
                                auto tmp8 = tmp4 * tmp7;
                                auto tmp11 = tmp5 * tmp10;
                                auto tmp12 = tmp9 - tmp11;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                tmp_acc1_vec = tmp_acc1_vec + tmp8;
                                tmp_acc2_vec = tmp_acc2_vec + tmp12;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    auto tmp9 = tmp8.neg();
                    auto tmp10 = tmp9 * tmp6;
                    auto tmp11 = tmp7 + tmp10;
                    tmp11.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr7[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            auto tmp6 = static_cast<float>(0.14433756729740643);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp8.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            auto tmp5 = static_cast<float>(0.14433756729740643);
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_82 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))] = tmp4;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(48L); x4+=static_cast<long>(1L))
                            {
                                auto tmp3 = in_ptr0[static_cast<long>(x1 + (196L*x4) + (9408L*x3) + (150528L*x0))];
                                auto tmp8 = in_ptr1[static_cast<long>(x4 + (48L*x1) + (9408L*x3) + (150528L*x0))];
                                auto tmp0 = c10::convert<int>(x2);
                                auto tmp1 = static_cast<int>(1);
                                auto tmp2 = tmp0 == tmp1;
                                auto tmp4 = static_cast<float>(0.0);
                                auto tmp5 = tmp2 ? tmp3 : tmp4;
                                auto tmp6 = static_cast<int>(0);
                                auto tmp7 = tmp0 == tmp6;
                                auto tmp9 = tmp7 ? tmp8 : tmp4;
                                auto tmp10 = decltype(tmp5)(tmp5 + tmp9);
                                out_ptr0[static_cast<long>(x4 + (48L*x3) + (768L*x2) + (1536L*x1) + (301056L*x0))] = tmp10;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_84 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp14 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 - tmp11;
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp12 - tmp16;
                    auto tmp18 = at::vec::Vectorized<float>(tmp1);
                    auto tmp19 = tmp18 * tmp17;
                    auto tmp20 = tmp0 + tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_gelu_backward_sum_85 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4816896L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_backward_sum_86 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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


cpp_fused_clone_sum_87 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr4 = in_out_ptr0;
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                                auto tmp1 = tmp0.neg();
                                auto tmp3 = decltype(tmp2)(1)/(decltype(tmp2)(1) + tmp2.neg().exp());
                                auto tmp4 = static_cast<float>(1.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp5 - tmp3;
                                auto tmp8 = tmp6 * tmp7;
                                auto tmp10 = tmp3 * tmp9;
                                auto tmp11 = tmp8 + tmp10;
                                auto tmp13 = tmp11 / tmp12;
                                auto tmp14 = tmp13 / tmp12;
                                auto tmp15 = tmp1 * tmp14;
                                tmp_acc0_vec = tmp_acc0_vec + tmp15;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp4 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x1)];
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp17 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp6 * tmp9;
                            auto tmp12 = tmp10 * tmp11;
                            auto tmp13 = static_cast<float>(1.0);
                            auto tmp14 = decltype(tmp13)(tmp13 - tmp8);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp6 * tmp15;
                            auto tmp18 = tmp16 * tmp17;
                            tmp12.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            tmp18.store(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr4[static_cast<long>(x1 + (16L*x2) + (3136L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x1)];
                            auto tmp8 = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp13 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp5)(1) / (decltype(tmp5)(1) + std::exp(-tmp5));
                            auto tmp7 = decltype(tmp4)(tmp4 * tmp6);
                            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp10)(tmp10 - tmp6);
                            auto tmp12 = decltype(tmp4)(tmp4 * tmp11);
                            auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                            out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp9;
                            out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp14;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (16L*x2) + (3136L*x1)));
                                auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr0[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr3[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x3) + (3136L*x2) + (614656L*x1)));
                                auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x0) + (38416L*x0_inner) + (614656L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = out_ptr3[static_cast<long>(x2 + (196L*x0) + (196L*x0_inner) + (3136L*x1))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                auto tmp6 = tmp4 * tmp5;
                                auto tmp8 = tmp4 * tmp7;
                                auto tmp11 = tmp5 * tmp10;
                                auto tmp12 = tmp9 - tmp11;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                                tmp_acc1_vec = tmp_acc1_vec + tmp8;
                                tmp_acc2_vec = tmp_acc2_vec + tmp12;
                            }
                        }
                    }
                    tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr5 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    auto tmp9 = tmp8.neg();
                    auto tmp10 = tmp9 * tmp6;
                    auto tmp11 = tmp7 + tmp10;
                    tmp11.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr7[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            auto tmp6 = static_cast<float>(0.14433756729740643);
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 * tmp7;
                            tmp8.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr2[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp2 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            auto tmp5 = static_cast<float>(0.14433756729740643);
                            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_90 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            auto tmp5 = tmp0 - tmp4;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp2 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = decltype(tmp0)(tmp0 - tmp3);
                            out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))] = tmp4;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(48L); x4+=static_cast<long>(1L))
                            {
                                auto tmp3 = in_ptr0[static_cast<long>(x1 + (196L*x4) + (9408L*x3) + (150528L*x0))];
                                auto tmp8 = in_ptr1[static_cast<long>(x4 + (48L*x1) + (9408L*x3) + (150528L*x0))];
                                auto tmp0 = c10::convert<int>(x2);
                                auto tmp1 = static_cast<int>(1);
                                auto tmp2 = tmp0 == tmp1;
                                auto tmp4 = static_cast<float>(0.0);
                                auto tmp5 = tmp2 ? tmp3 : tmp4;
                                auto tmp6 = static_cast<int>(0);
                                auto tmp7 = tmp0 == tmp6;
                                auto tmp9 = tmp7 ? tmp8 : tmp4;
                                auto tmp10 = decltype(tmp5)(tmp5 + tmp9);
                                out_ptr0[static_cast<long>(x4 + (48L*x3) + (768L*x2) + (1536L*x1) + (301056L*x0))] = tmp10;
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_backward_sum_92 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
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
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp2;
                    }
                    tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp14 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 * tmp5;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 - tmp11;
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp12 - tmp16;
                    auto tmp18 = at::vec::Vectorized<float>(tmp1);
                    auto tmp19 = tmp18 * tmp17;
                    auto tmp20 = tmp0 + tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (151296L*x1)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x0));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(150528L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (150528L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_3, primals_5, primals_6, primals_8, primals_10, primals_11, primals_13, primals_15, primals_16, primals_18, primals_20, primals_21, primals_23, primals_25, primals_26, primals_28, primals_30, primals_31, primals_33, primals_35, primals_36, primals_38, primals_40, primals_41, primals_43, primals_45, primals_46, primals_48, primals_50, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_181, mul, view_5, view_8, div, div_1, unsqueeze_5, view_21, mul_5, view_23, addmm_1, view_25, mul_10, view_31, div_3, div_4, unsqueeze_11, view_47, mul_15, view_49, addmm_4, view_51, mul_20, view_57, div_6, div_7, unsqueeze_17, view_73, mul_25, view_75, addmm_7, view_77, mul_30, view_83, div_9, div_10, unsqueeze_23, view_99, mul_35, view_101, addmm_10, view_103, mul_40, view_109, div_12, div_13, unsqueeze_29, view_125, mul_45, view_127, addmm_13, view_129, mul_50, view_135, div_15, div_16, unsqueeze_35, view_151, mul_55, view_153, addmm_16, view_155, mul_60, view_161, div_18, div_19, unsqueeze_41, view_177, mul_65, view_179, addmm_19, view_181, mul_70, view_187, div_21, div_22, unsqueeze_47, view_203, mul_75, view_205, addmm_22, view_207, mul_80, view_213, div_24, div_25, unsqueeze_53, view_229, mul_85, view_231, addmm_25, view_233, mul_90, view_239, div_27, div_28, unsqueeze_59, view_255, mul_95, view_257, addmm_28, view_259, cat, getitem_41, rsqrt_20, view_261, view_271, mul_103, view_273, addmm_31, view_275, mul_108, view_277, view_287, mul_111, view_289, addmm_34, view_291, mul_116, clone_167, permute_126, div_32, permute_130, permute_134, div_33, permute_138, permute_143, permute_144, alias_42, permute_145, permute_146, permute_151, div_34, permute_153, permute_157, div_35, permute_161, permute_166, permute_167, alias_43, permute_168, permute_169, permute_174, permute_176, permute_180, div_37, permute_184, permute_189, permute_190, permute_194, permute_196, permute_197, permute_206, div_41, permute_208, permute_212, div_42, permute_216, permute_221, permute_222, permute_226, permute_228, permute_229, permute_238, div_46, permute_240, permute_244, div_47, permute_248, permute_253, permute_254, permute_258, permute_260, permute_261, permute_270, div_51, permute_272, permute_276, div_52, permute_280, permute_285, permute_286, permute_290, permute_292, permute_293, permute_302, div_56, permute_304, permute_308, div_57, permute_312, permute_317, permute_318, permute_322, permute_324, permute_325, permute_334, div_61, permute_336, permute_340, div_62, permute_344, permute_349, permute_350, permute_354, permute_356, permute_357, permute_366, div_66, permute_368, permute_372, div_67, permute_376, permute_381, permute_382, permute_386, permute_388, permute_389, permute_398, div_71, permute_400, permute_404, div_72, permute_408, permute_413, permute_414, permute_418, permute_420, permute_421, permute_430, div_76, permute_432, permute_436, div_77, permute_440, permute_445, permute_446, permute_450, permute_452, permute_453, permute_462, div_81, permute_464, permute_468, div_82, permute_472, permute_477, permute_478, permute_482, permute_484, permute_485, permute_494, div_86, tangents_1 = args
    args.clear()
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_10, (16, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_15, (16, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_20, (16, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_25, (16, ), (1, ))
    assert_size_stride(primals_26, (768, ), (1, ))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_30, (16, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_35, (16, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_40, (16, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_45, (16, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_50, (16, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_63, (768, 3, 16, 16), (768, 1, 48, 3))
    assert_size_stride(primals_181, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(mul, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_5, (1568, 768), (768, 1))
    assert_size_stride(view_8, (307328, 3), (3, 1))
    assert_size_stride(div, (8, 16, 196, 196), (614656, 1, 3136, 16))
    assert_size_stride(div_1, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_5, (8, 16, 196, 1), (3136, 1, 16, 16))
    assert_size_stride(view_21, (1568, 768), (768, 1))
    assert_size_stride(mul_5, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_23, (1568, 768), (768, 1))
    assert_size_stride(addmm_1, (1568, 3072), (3072, 1))
    assert_size_stride(view_25, (1568, 3072), (3072, 1))
    assert_size_stride(mul_10, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_31, (1568, 768), (768, 1))
    assert_size_stride(div_3, (8, 16, 196, 196), (614656, 1, 3136, 16))
    assert_size_stride(div_4, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_11, (8, 16, 196, 1), (3136, 1, 16, 16))
    assert_size_stride(view_47, (1568, 768), (768, 1))
    assert_size_stride(mul_15, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_49, (1568, 768), (768, 1))
    assert_size_stride(addmm_4, (1568, 3072), (3072, 1))
    assert_size_stride(view_51, (1568, 3072), (3072, 1))
    assert_size_stride(mul_20, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_57, (1568, 768), (768, 1))
    assert_size_stride(div_6, (8, 16, 196, 196), (614656, 1, 3136, 16))
    assert_size_stride(div_7, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_17, (8, 16, 196, 1), (3136, 1, 16, 16))
    assert_size_stride(view_73, (1568, 768), (768, 1))
    assert_size_stride(mul_25, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_75, (1568, 768), (768, 1))
    assert_size_stride(addmm_7, (1568, 3072), (3072, 1))
    assert_size_stride(view_77, (1568, 3072), (3072, 1))
    assert_size_stride(mul_30, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_83, (1568, 768), (768, 1))
    assert_size_stride(div_9, (8, 16, 196, 196), (614656, 1, 3136, 16))
    assert_size_stride(div_10, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_23, (8, 16, 196, 1), (3136, 1, 16, 16))
    assert_size_stride(view_99, (1568, 768), (768, 1))
    assert_size_stride(mul_35, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_101, (1568, 768), (768, 1))
    assert_size_stride(addmm_10, (1568, 3072), (3072, 1))
    assert_size_stride(view_103, (1568, 3072), (3072, 1))
    assert_size_stride(mul_40, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_109, (1568, 768), (768, 1))
    assert_size_stride(div_12, (8, 16, 196, 196), (614656, 1, 3136, 16))
    assert_size_stride(div_13, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_29, (8, 16, 196, 1), (3136, 1, 16, 16))
    assert_size_stride(view_125, (1568, 768), (768, 1))
    assert_size_stride(mul_45, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_127, (1568, 768), (768, 1))
    assert_size_stride(addmm_13, (1568, 3072), (3072, 1))
    assert_size_stride(view_129, (1568, 3072), (3072, 1))
    assert_size_stride(mul_50, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_135, (1568, 768), (768, 1))
    assert_size_stride(div_15, (8, 16, 196, 196), (614656, 1, 3136, 16))
    assert_size_stride(div_16, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_35, (8, 16, 196, 1), (3136, 1, 16, 16))
    assert_size_stride(view_151, (1568, 768), (768, 1))
    assert_size_stride(mul_55, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_153, (1568, 768), (768, 1))
    assert_size_stride(addmm_16, (1568, 3072), (3072, 1))
    assert_size_stride(view_155, (1568, 3072), (3072, 1))
    assert_size_stride(mul_60, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_161, (1568, 768), (768, 1))
    assert_size_stride(div_18, (8, 16, 196, 196), (614656, 1, 3136, 16))
    assert_size_stride(div_19, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_41, (8, 16, 196, 1), (3136, 1, 16, 16))
    assert_size_stride(view_177, (1568, 768), (768, 1))
    assert_size_stride(mul_65, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_179, (1568, 768), (768, 1))
    assert_size_stride(addmm_19, (1568, 3072), (3072, 1))
    assert_size_stride(view_181, (1568, 3072), (3072, 1))
    assert_size_stride(mul_70, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_187, (1568, 768), (768, 1))
    assert_size_stride(div_21, (8, 16, 196, 196), (614656, 1, 3136, 16))
    assert_size_stride(div_22, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_47, (8, 16, 196, 1), (3136, 1, 16, 16))
    assert_size_stride(view_203, (1568, 768), (768, 1))
    assert_size_stride(mul_75, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_205, (1568, 768), (768, 1))
    assert_size_stride(addmm_22, (1568, 3072), (3072, 1))
    assert_size_stride(view_207, (1568, 3072), (3072, 1))
    assert_size_stride(mul_80, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_213, (1568, 768), (768, 1))
    assert_size_stride(div_24, (8, 16, 196, 196), (614656, 1, 3136, 16))
    assert_size_stride(div_25, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_53, (8, 16, 196, 1), (3136, 1, 16, 16))
    assert_size_stride(view_229, (1568, 768), (768, 1))
    assert_size_stride(mul_85, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_231, (1568, 768), (768, 1))
    assert_size_stride(addmm_25, (1568, 3072), (3072, 1))
    assert_size_stride(view_233, (1568, 3072), (3072, 1))
    assert_size_stride(mul_90, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_239, (1568, 768), (768, 1))
    assert_size_stride(div_27, (8, 16, 196, 196), (614656, 1, 3136, 16))
    assert_size_stride(div_28, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_59, (8, 16, 196, 1), (3136, 1, 16, 16))
    assert_size_stride(view_255, (1568, 768), (768, 1))
    assert_size_stride(mul_95, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_257, (1568, 768), (768, 1))
    assert_size_stride(addmm_28, (1568, 3072), (3072, 1))
    assert_size_stride(view_259, (1568, 3072), (3072, 1))
    assert_size_stride(cat, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(getitem_41, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_20, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_261, (1576, 768), (768, 1))
    assert_size_stride(view_271, (1576, 768), (768, 1))
    assert_size_stride(mul_103, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_273, (1576, 768), (768, 1))
    assert_size_stride(addmm_31, (1576, 3072), (3072, 1))
    assert_size_stride(view_275, (1576, 3072), (3072, 1))
    assert_size_stride(mul_108, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_277, (1576, 768), (768, 1))
    assert_size_stride(view_287, (1576, 768), (768, 1))
    assert_size_stride(mul_111, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_289, (1576, 768), (768, 1))
    assert_size_stride(addmm_34, (1576, 3072), (3072, 1))
    assert_size_stride(view_291, (1576, 3072), (3072, 1))
    assert_size_stride(mul_116, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(clone_167, (8, 768), (768, 1))
    assert_size_stride(permute_126, (1000, 768), (768, 1))
    assert_size_stride(div_32, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_130, (768, 3072), (3072, 1))
    assert_size_stride(permute_134, (3072, 768), (768, 1))
    assert_size_stride(div_33, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_138, (768, 768), (768, 1))
    assert_size_stride(permute_143, (128, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_144, (128, 48, 197), (9456, 1, 48))
    assert_size_stride(alias_42, (8, 16, 197, 197), (620944, 1, 3152, 16))
    assert_size_stride(permute_145, (128, 48, 197), (9456, 1, 48))
    assert_size_stride(permute_146, (128, 197, 48), (9456, 1, 197))
    assert_size_stride(permute_151, (2304, 768), (768, 1))
    assert_size_stride(div_34, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_153, (768, 3072), (3072, 1))
    assert_size_stride(permute_157, (3072, 768), (768, 1))
    assert_size_stride(div_35, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_161, (768, 768), (768, 1))
    assert_size_stride(permute_166, (128, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_167, (128, 48, 197), (9456, 1, 48))
    assert_size_stride(alias_43, (8, 16, 197, 197), (620944, 1, 3152, 16))
    assert_size_stride(permute_168, (128, 48, 197), (9456, 1, 48))
    assert_size_stride(permute_169, (128, 197, 48), (9456, 1, 197))
    assert_size_stride(permute_174, (2304, 768), (768, 1))
    assert_size_stride(permute_176, (768, 3072), (3072, 1))
    assert_size_stride(permute_180, (3072, 768), (768, 1))
    assert_size_stride(div_37, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_184, (768, 768), (768, 1))
    assert_size_stride(permute_189, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_190, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_194, (768, 768), (768, 1))
    assert_size_stride(permute_196, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_197, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_206, (1536, 768), (768, 1))
    assert_size_stride(div_41, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_208, (768, 3072), (3072, 1))
    assert_size_stride(permute_212, (3072, 768), (768, 1))
    assert_size_stride(div_42, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_216, (768, 768), (768, 1))
    assert_size_stride(permute_221, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_222, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_226, (768, 768), (768, 1))
    assert_size_stride(permute_228, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_229, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_238, (1536, 768), (768, 1))
    assert_size_stride(div_46, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_240, (768, 3072), (3072, 1))
    assert_size_stride(permute_244, (3072, 768), (768, 1))
    assert_size_stride(div_47, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_248, (768, 768), (768, 1))
    assert_size_stride(permute_253, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_254, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_258, (768, 768), (768, 1))
    assert_size_stride(permute_260, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_261, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_270, (1536, 768), (768, 1))
    assert_size_stride(div_51, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_272, (768, 3072), (3072, 1))
    assert_size_stride(permute_276, (3072, 768), (768, 1))
    assert_size_stride(div_52, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_280, (768, 768), (768, 1))
    assert_size_stride(permute_285, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_286, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_290, (768, 768), (768, 1))
    assert_size_stride(permute_292, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_293, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_302, (1536, 768), (768, 1))
    assert_size_stride(div_56, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_304, (768, 3072), (3072, 1))
    assert_size_stride(permute_308, (3072, 768), (768, 1))
    assert_size_stride(div_57, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_312, (768, 768), (768, 1))
    assert_size_stride(permute_317, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_318, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_322, (768, 768), (768, 1))
    assert_size_stride(permute_324, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_325, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_334, (1536, 768), (768, 1))
    assert_size_stride(div_61, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_336, (768, 3072), (3072, 1))
    assert_size_stride(permute_340, (3072, 768), (768, 1))
    assert_size_stride(div_62, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_344, (768, 768), (768, 1))
    assert_size_stride(permute_349, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_350, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_354, (768, 768), (768, 1))
    assert_size_stride(permute_356, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_357, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_366, (1536, 768), (768, 1))
    assert_size_stride(div_66, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_368, (768, 3072), (3072, 1))
    assert_size_stride(permute_372, (3072, 768), (768, 1))
    assert_size_stride(div_67, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_376, (768, 768), (768, 1))
    assert_size_stride(permute_381, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_382, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_386, (768, 768), (768, 1))
    assert_size_stride(permute_388, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_389, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_398, (1536, 768), (768, 1))
    assert_size_stride(div_71, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_400, (768, 3072), (3072, 1))
    assert_size_stride(permute_404, (3072, 768), (768, 1))
    assert_size_stride(div_72, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_408, (768, 768), (768, 1))
    assert_size_stride(permute_413, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_414, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_418, (768, 768), (768, 1))
    assert_size_stride(permute_420, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_421, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_430, (1536, 768), (768, 1))
    assert_size_stride(div_76, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_432, (768, 3072), (3072, 1))
    assert_size_stride(permute_436, (3072, 768), (768, 1))
    assert_size_stride(div_77, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_440, (768, 768), (768, 1))
    assert_size_stride(permute_445, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_446, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_450, (768, 768), (768, 1))
    assert_size_stride(permute_452, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_453, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_462, (1536, 768), (768, 1))
    assert_size_stride(div_81, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_464, (768, 3072), (3072, 1))
    assert_size_stride(permute_468, (3072, 768), (768, 1))
    assert_size_stride(div_82, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_472, (768, 768), (768, 1))
    assert_size_stride(permute_477, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_478, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_482, (768, 768), (768, 1))
    assert_size_stride(permute_484, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_485, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_494, (1536, 768), (768, 1))
    assert_size_stride(div_86, (8, 196, 1), (196, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_126, out=buf0)
    del permute_126
    buf1 = empty((1000, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_167, out=buf1)
    del clone_167
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf5 = empty((8, 197, 768), device='cpu', dtype=torch.float32)
    buf6 = empty((768, ), device='cpu', dtype=torch.float32)
    buf7 = empty((768, ), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_backward_select_backward_sum_0(c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(mul_116.data_ptr()), c_void_p(div_32.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del buf0
    del div_32
    del mul_116
    del primals_61
    del tangents_1
    buf8 = empty((1576, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (1576, 768), (768, 1), 0), permute_130, out=buf8)
    del permute_130
    buf9 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf5, (768, 1576), (1, 768), 0), view_291, out=buf9)
    del view_291
    buf10 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf11 = reinterpret_tensor(buf8, (8, 197, 3072), (605184, 3072, 1), 0); del buf8  # reuse
    cpp_fused_gelu_gelu_backward_sum_1(c_void_p(buf11.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(addmm_34.data_ptr()), c_void_p(buf10.data_ptr()))
    del addmm_34
    buf12 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (1576, 3072), (3072, 1), 0), permute_134, out=buf12)
    del permute_134
    buf13 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf11, (3072, 1576), (1, 3072), 0), view_289, out=buf13)
    del view_289
    buf14 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf15 = buf4; del buf4  # reuse
    buf16 = buf3; del buf3  # reuse
    buf17 = empty((768, ), device='cpu', dtype=torch.float32)
    buf18 = empty((768, ), device='cpu', dtype=torch.float32)
    buf19 = reinterpret_tensor(buf12, (8, 197, 768), (151296, 768, 1), 0); del buf12  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_2(c_void_p(buf19.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(mul_111.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(div_33.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()))
    del div_33
    del mul_111
    del primals_59
    buf20 = reinterpret_tensor(buf5, (1576, 768), (768, 1), 0); del buf5  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf19, (1576, 768), (768, 1), 0), permute_138, out=buf20)
    del permute_138
    buf21 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf19, (768, 1576), (1, 768), 0), view_287, out=buf21)
    del view_287
    buf22 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf23 = empty((8, 16, 197, 48), device='cpu', dtype=torch.float32)
    cpp_fused_clone_sum_3(c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()))
    buf24 = reinterpret_tensor(buf20, (128, 197, 48), (9456, 48, 1), 0); del buf20  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_143, reinterpret_tensor(buf23, (128, 197, 48), (9456, 48, 1), 0), out=buf24)
    del permute_143
    buf25 = empty((128, 197, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf23, (128, 197, 48), (9456, 48, 1), 0), permute_144, out=buf25)
    del permute_144
    buf26 = empty_strided((8, 16, 197, 1), (3152, 197, 1, 25216), device='cpu', dtype=torch.float32)
    buf27 = reinterpret_tensor(buf25, (8, 16, 197, 197), (620944, 38809, 197, 1), 0); del buf25  # reuse
    cpp_fused__softmax_backward_data_mul_4(c_void_p(buf27.data_ptr()), c_void_p(alias_42.data_ptr()), c_void_p(buf26.data_ptr()))
    del alias_42
    buf28 = reinterpret_tensor(buf23, (128, 48, 197), (9456, 197, 1), 0); del buf23  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_145, reinterpret_tensor(buf27, (128, 197, 197), (38809, 197, 1), 0), out=buf28)
    del permute_145
    buf29 = empty((128, 197, 48), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf27, (128, 197, 197), (38809, 197, 1), 0), permute_146, out=buf29)
    del permute_146
    buf30 = empty((8, 197, 3, 16, 48), device='cpu', dtype=torch.float32)
    cpp_fused_clone_5(c_void_p(buf29.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf30.data_ptr()))
    buf31 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf30, (2304, 1576), (1, 2304), 0), view_277, out=buf31)
    del view_277
    buf32 = reinterpret_tensor(buf29, (1576, 768), (768, 1), 0); del buf29  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf30, (1576, 2304), (2304, 1), 0), permute_151, out=buf32)
    del permute_151
    buf33 = buf16; del buf16  # reuse
    buf34 = buf15; del buf15  # reuse
    buf35 = empty((768, ), device='cpu', dtype=torch.float32)
    buf36 = empty((768, ), device='cpu', dtype=torch.float32)
    buf37 = buf19; del buf19  # reuse
    cpp_fused_add_native_layer_norm_backward_6(c_void_p(buf37.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(mul_108.data_ptr()), c_void_p(div_34.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()))
    del div_34
    del mul_108
    del primals_57
    buf38 = reinterpret_tensor(buf11, (1576, 3072), (3072, 1), 0); del buf11  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf37, (1576, 768), (768, 1), 0), permute_153, out=buf38)
    del permute_153
    buf39 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf37, (768, 1576), (1, 768), 0), view_275, out=buf39)
    del view_275
    buf40 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf41 = reinterpret_tensor(buf38, (8, 197, 3072), (605184, 3072, 1), 0); del buf38  # reuse
    cpp_fused_gelu_gelu_backward_sum_7(c_void_p(buf41.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(addmm_31.data_ptr()), c_void_p(buf40.data_ptr()))
    del addmm_31
    buf42 = buf32; del buf32  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf41, (1576, 3072), (3072, 1), 0), permute_157, out=buf42)
    del permute_157
    buf43 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf41, (3072, 1576), (1, 3072), 0), view_273, out=buf43)
    del view_273
    buf44 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf45 = buf34; del buf34  # reuse
    buf46 = buf33; del buf33  # reuse
    buf47 = empty((768, ), device='cpu', dtype=torch.float32)
    buf48 = empty((768, ), device='cpu', dtype=torch.float32)
    buf49 = buf37; del buf37  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_8(c_void_p(buf49.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(mul_103.data_ptr()), c_void_p(div_35.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    del buf41
    del div_35
    del mul_103
    del primals_55
    buf50 = buf42; del buf42  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf49, (1576, 768), (768, 1), 0), permute_161, out=buf50)
    del permute_161
    buf51 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf49, (768, 1576), (1, 768), 0), view_271, out=buf51)
    del view_271
    buf52 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf53 = reinterpret_tensor(buf28, (8, 16, 197, 48), (151296, 9456, 48, 1), 0); del buf28  # reuse
    cpp_fused_clone_sum_9(c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()))
    buf54 = reinterpret_tensor(buf50, (128, 197, 48), (9456, 48, 1), 0); del buf50  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_166, reinterpret_tensor(buf53, (128, 197, 48), (9456, 48, 1), 0), out=buf54)
    del permute_166
    buf55 = reinterpret_tensor(buf27, (128, 197, 197), (38809, 197, 1), 0); del buf27  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf53, (128, 197, 48), (9456, 48, 1), 0), permute_167, out=buf55)
    del permute_167
    buf56 = buf26; del buf26  # reuse
    buf57 = reinterpret_tensor(buf55, (8, 16, 197, 197), (620944, 38809, 197, 1), 0); del buf55  # reuse
    cpp_fused__softmax_backward_data_mul_10(c_void_p(buf57.data_ptr()), c_void_p(alias_43.data_ptr()), c_void_p(buf56.data_ptr()))
    del alias_43
    del buf56
    buf58 = reinterpret_tensor(buf53, (128, 48, 197), (9456, 197, 1), 0); del buf53  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_168, reinterpret_tensor(buf57, (128, 197, 197), (38809, 197, 1), 0), out=buf58)
    del permute_168
    buf59 = buf24; del buf24  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf57, (128, 197, 197), (38809, 197, 1), 0), permute_169, out=buf59)
    del buf57
    del permute_169
    buf60 = buf30; del buf30  # reuse
    cpp_fused_clone_11(c_void_p(buf59.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf60.data_ptr()))
    del buf54
    del buf58
    buf61 = empty((2304, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf60, (2304, 1576), (1, 2304), 0), view_261, out=buf61)
    del view_261
    buf62 = reinterpret_tensor(buf59, (1576, 768), (768, 1), 0); del buf59  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf60, (1576, 2304), (2304, 1), 0), permute_174, out=buf62)
    del buf60
    del permute_174
    buf63 = buf46; del buf46  # reuse
    buf64 = buf45; del buf45  # reuse
    buf65 = empty((768, ), device='cpu', dtype=torch.float32)
    buf66 = empty((768, ), device='cpu', dtype=torch.float32)
    buf67 = buf49; del buf49  # reuse
    buf68 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_add_clone_native_layer_norm_native_layer_norm_backward_12(c_void_p(buf67.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(cat.data_ptr()), c_void_p(getitem_41.data_ptr()), c_void_p(rsqrt_20.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf68.data_ptr()))
    del buf62
    del buf63
    del buf64
    del cat
    del getitem_41
    del primals_53
    del rsqrt_20
    buf69 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf68, permute_176, out=buf69)
    del permute_176
    buf70 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf68, (768, 1568), (1, 768), 0), view_259, out=buf70)
    del view_259
    buf71 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf72 = reinterpret_tensor(buf69, (8, 196, 3072), (602112, 3072, 1), 0); del buf69  # reuse
    cpp_fused_gelu_gelu_backward_sum_13(c_void_p(buf72.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(addmm_28.data_ptr()), c_void_p(buf71.data_ptr()))
    del addmm_28
    buf73 = buf68; del buf68  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf72, (1568, 3072), (3072, 1), 0), permute_180, out=buf73)
    del permute_180
    buf74 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf72, (3072, 1568), (1, 3072), 0), view_257, out=buf74)
    del view_257
    buf75 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf76 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf77 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf78 = empty((768, ), device='cpu', dtype=torch.float32)
    buf79 = empty((768, ), device='cpu', dtype=torch.float32)
    buf80 = reinterpret_tensor(buf73, (8, 196, 768), (150528, 768, 1), 0); del buf73  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_14(c_void_p(buf80.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(mul_95.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(div_37.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    del div_37
    del mul_95
    del primals_51
    buf81 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf80, (1568, 768), (768, 1), 0), permute_184, out=buf81)
    del permute_184
    buf82 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf80, (768, 1568), (1, 768), 0), view_255, out=buf82)
    del view_255
    buf83 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf84 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    cpp_fused_clone_sum_15(c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()))
    buf85 = reinterpret_tensor(buf81, (128, 196, 48), (9408, 48, 1), 0); del buf81  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_189, reinterpret_tensor(buf84, (128, 196, 48), (9408, 48, 1), 0), out=buf85)
    del permute_189
    buf86 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf84, (128, 196, 48), (9408, 48, 1), 0), permute_190, out=buf86)
    del permute_190
    buf87 = reinterpret_tensor(buf84, (1568, 768), (768, 1), 0); del buf84  # reuse
    cpp_fused_view_16(c_void_p(buf85.data_ptr()), c_void_p(buf87.data_ptr()))
    buf88 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf87, (768, 1568), (1, 768), 0), view_239, out=buf88)
    buf89 = reinterpret_tensor(buf85, (1568, 768), (768, 1), 0); del buf85  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf87, permute_194, out=buf89)
    del permute_194
    buf90 = empty((8, 16, 196, 1), device='cpu', dtype=torch.float32)
    buf94 = empty((8, 16, 196, 196), device='cpu', dtype=torch.float32)
    buf96 = empty((8, 16, 196, 196), device='cpu', dtype=torch.float32)
    buf95 = empty_strided((8, 16, 196, 1), (3136, 196, 1, 25088), device='cpu', dtype=torch.float32)
    buf91 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf92 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf101 = empty((1, 1, 1, 16), device='cpu', dtype=torch.float32)
    buf93 = reinterpret_tensor(buf91, (16, ), (1, ), 0); del buf91  # reuse
    buf97 = empty_strided((8, 16, 196, 1), (3136, 196, 1, 25088), device='cpu', dtype=torch.float32)
    buf98 = buf96; del buf96  # reuse
    cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_17(c_void_p(buf93.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(div_27.data_ptr()), c_void_p(div_28.data_ptr()), c_void_p(unsqueeze_59.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf97.data_ptr()))
    del div_27
    del primals_50
    del unsqueeze_59
    buf99 = reinterpret_tensor(buf87, (128, 48, 196), (9408, 196, 1), 0); del buf87  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_196, reinterpret_tensor(buf98, (128, 196, 196), (38416, 196, 1), 0), out=buf99)
    del permute_196
    buf100 = empty((128, 196, 48), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf98, (128, 196, 196), (38416, 196, 1), 0), permute_197, out=buf100)
    del permute_197
    buf102 = reinterpret_tensor(buf98, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf98  # reuse
    cpp_fused_clone_18(c_void_p(buf94.data_ptr()), c_void_p(div_28.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf102.data_ptr()))
    del div_28
    buf103 = empty((16, 3), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf102, (16, 307328), (1, 16), 0), view_8, out=buf103)
    buf104 = empty((8, 196, 2, 16, 48), device='cpu', dtype=torch.float32)
    cpp_fused_clone_19(c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf104.data_ptr()))
    buf105 = empty((1536, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (1536, 1568), (1, 1536), 0), view_239, out=buf105)
    del view_239
    buf106 = reinterpret_tensor(buf99, (1568, 768), (768, 1), 0); del buf99  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (1568, 1536), (1536, 1), 0), permute_206, out=buf106)
    del permute_206
    buf107 = buf77; del buf77  # reuse
    buf108 = buf76; del buf76  # reuse
    buf109 = empty((768, ), device='cpu', dtype=torch.float32)
    buf110 = empty((768, ), device='cpu', dtype=torch.float32)
    buf111 = reinterpret_tensor(buf106, (8, 196, 768), (150528, 768, 1), 0); del buf106  # reuse
    cpp_fused_add_native_layer_norm_backward_20(c_void_p(buf111.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(mul_90.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(div_41.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()))
    del div_41
    del mul_90
    del primals_48
    buf112 = reinterpret_tensor(buf72, (1568, 3072), (3072, 1), 0); del buf72  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf111, (1568, 768), (768, 1), 0), permute_208, out=buf112)
    del permute_208
    buf113 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf111, (768, 1568), (1, 768), 0), view_233, out=buf113)
    del view_233
    buf114 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf115 = reinterpret_tensor(buf112, (8, 196, 3072), (602112, 3072, 1), 0); del buf112  # reuse
    cpp_fused_gelu_gelu_backward_sum_21(c_void_p(buf115.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(addmm_25.data_ptr()), c_void_p(buf114.data_ptr()))
    del addmm_25
    buf116 = buf89; del buf89  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf115, (1568, 3072), (3072, 1), 0), permute_212, out=buf116)
    del permute_212
    buf117 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf115, (3072, 1568), (1, 3072), 0), view_231, out=buf117)
    del view_231
    buf118 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf119 = buf108; del buf108  # reuse
    buf120 = buf107; del buf107  # reuse
    buf121 = empty((768, ), device='cpu', dtype=torch.float32)
    buf122 = empty((768, ), device='cpu', dtype=torch.float32)
    buf123 = buf111; del buf111  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_22(c_void_p(buf123.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(mul_85.data_ptr()), c_void_p(div_42.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()))
    del div_42
    del mul_85
    del primals_46
    buf124 = buf116; del buf116  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf123, (1568, 768), (768, 1), 0), permute_216, out=buf124)
    del permute_216
    buf125 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf123, (768, 1568), (1, 768), 0), view_229, out=buf125)
    del view_229
    buf126 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf127 = reinterpret_tensor(buf80, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf80  # reuse
    cpp_fused_clone_sum_23(c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()))
    buf128 = reinterpret_tensor(buf124, (128, 196, 48), (9408, 48, 1), 0); del buf124  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_221, reinterpret_tensor(buf127, (128, 196, 48), (9408, 48, 1), 0), out=buf128)
    del permute_221
    buf129 = reinterpret_tensor(buf102, (128, 196, 196), (38416, 196, 1), 0); del buf102  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf127, (128, 196, 48), (9408, 48, 1), 0), permute_222, out=buf129)
    del permute_222
    buf130 = reinterpret_tensor(buf127, (1568, 768), (768, 1), 0); del buf127  # reuse
    cpp_fused_view_24(c_void_p(buf128.data_ptr()), c_void_p(buf130.data_ptr()))
    buf131 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf130, (768, 1568), (1, 768), 0), view_213, out=buf131)
    buf132 = reinterpret_tensor(buf128, (1568, 768), (768, 1), 0); del buf128  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf130, permute_226, out=buf132)
    del permute_226
    buf133 = reinterpret_tensor(buf95, (8, 16, 196, 1), (3136, 196, 1, 1), 0); del buf95  # reuse
    buf137 = buf94; del buf94  # reuse
    buf139 = reinterpret_tensor(buf86, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf86  # reuse
    buf138 = buf97; del buf97  # reuse
    buf134 = buf92; del buf92  # reuse
    buf135 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf144 = empty((1, 1, 1, 16), device='cpu', dtype=torch.float32)
    buf136 = reinterpret_tensor(buf134, (16, ), (1, ), 0); del buf134  # reuse
    buf140 = reinterpret_tensor(buf90, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf90  # reuse
    buf141 = buf139; del buf139  # reuse
    cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_25(c_void_p(buf136.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(div_24.data_ptr()), c_void_p(div_25.data_ptr()), c_void_p(unsqueeze_53.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf140.data_ptr()))
    del div_24
    del primals_45
    del unsqueeze_53
    buf142 = reinterpret_tensor(buf130, (128, 48, 196), (9408, 196, 1), 0); del buf130  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_228, reinterpret_tensor(buf141, (128, 196, 196), (38416, 196, 1), 0), out=buf142)
    del permute_228
    buf143 = buf100; del buf100  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf141, (128, 196, 196), (38416, 196, 1), 0), permute_229, out=buf143)
    del permute_229
    buf145 = reinterpret_tensor(buf141, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf141  # reuse
    cpp_fused_clone_26(c_void_p(buf137.data_ptr()), c_void_p(div_25.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf145.data_ptr()))
    del div_25
    buf146 = empty((16, 3), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (16, 307328), (1, 16), 0), view_8, out=buf146)
    buf147 = buf104; del buf104  # reuse
    cpp_fused_clone_27(c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf147.data_ptr()))
    buf148 = empty((1536, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf147, (1536, 1568), (1, 1536), 0), view_213, out=buf148)
    del view_213
    buf149 = reinterpret_tensor(buf143, (1568, 768), (768, 1), 0); del buf143  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf147, (1568, 1536), (1536, 1), 0), permute_238, out=buf149)
    del permute_238
    buf150 = buf120; del buf120  # reuse
    buf151 = buf119; del buf119  # reuse
    buf152 = empty((768, ), device='cpu', dtype=torch.float32)
    buf153 = empty((768, ), device='cpu', dtype=torch.float32)
    buf154 = buf123; del buf123  # reuse
    cpp_fused_add_native_layer_norm_backward_28(c_void_p(buf154.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(mul_80.data_ptr()), c_void_p(div_46.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()))
    del div_46
    del mul_80
    del primals_43
    buf155 = reinterpret_tensor(buf115, (1568, 3072), (3072, 1), 0); del buf115  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf154, (1568, 768), (768, 1), 0), permute_240, out=buf155)
    del permute_240
    buf156 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf154, (768, 1568), (1, 768), 0), view_207, out=buf156)
    del view_207
    buf157 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf158 = reinterpret_tensor(buf155, (8, 196, 3072), (602112, 3072, 1), 0); del buf155  # reuse
    cpp_fused_gelu_gelu_backward_sum_29(c_void_p(buf158.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(addmm_22.data_ptr()), c_void_p(buf157.data_ptr()))
    del addmm_22
    buf159 = buf149; del buf149  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf158, (1568, 3072), (3072, 1), 0), permute_244, out=buf159)
    del permute_244
    buf160 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf158, (3072, 1568), (1, 3072), 0), view_205, out=buf160)
    del view_205
    buf161 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf162 = buf151; del buf151  # reuse
    buf163 = buf150; del buf150  # reuse
    buf164 = empty((768, ), device='cpu', dtype=torch.float32)
    buf165 = empty((768, ), device='cpu', dtype=torch.float32)
    buf166 = buf154; del buf154  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_30(c_void_p(buf166.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(mul_75.data_ptr()), c_void_p(div_47.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()))
    del div_47
    del mul_75
    del primals_41
    buf167 = buf159; del buf159  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf166, (1568, 768), (768, 1), 0), permute_248, out=buf167)
    del permute_248
    buf168 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf166, (768, 1568), (1, 768), 0), view_203, out=buf168)
    del view_203
    buf169 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf170 = reinterpret_tensor(buf132, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf132  # reuse
    cpp_fused_clone_sum_31(c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()))
    buf171 = reinterpret_tensor(buf167, (128, 196, 48), (9408, 48, 1), 0); del buf167  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_253, reinterpret_tensor(buf170, (128, 196, 48), (9408, 48, 1), 0), out=buf171)
    del permute_253
    buf172 = reinterpret_tensor(buf145, (128, 196, 196), (38416, 196, 1), 0); del buf145  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf170, (128, 196, 48), (9408, 48, 1), 0), permute_254, out=buf172)
    del permute_254
    buf173 = reinterpret_tensor(buf170, (1568, 768), (768, 1), 0); del buf170  # reuse
    cpp_fused_view_32(c_void_p(buf171.data_ptr()), c_void_p(buf173.data_ptr()))
    buf174 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf173, (768, 1568), (1, 768), 0), view_187, out=buf174)
    buf175 = reinterpret_tensor(buf171, (1568, 768), (768, 1), 0); del buf171  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf173, permute_258, out=buf175)
    del permute_258
    buf176 = reinterpret_tensor(buf138, (8, 16, 196, 1), (3136, 196, 1, 1), 0); del buf138  # reuse
    buf180 = buf137; del buf137  # reuse
    buf182 = reinterpret_tensor(buf129, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf129  # reuse
    buf181 = buf140; del buf140  # reuse
    buf177 = buf135; del buf135  # reuse
    buf178 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf187 = empty((1, 1, 1, 16), device='cpu', dtype=torch.float32)
    buf179 = reinterpret_tensor(buf177, (16, ), (1, ), 0); del buf177  # reuse
    buf183 = reinterpret_tensor(buf133, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf133  # reuse
    buf184 = buf182; del buf182  # reuse
    cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_33(c_void_p(buf179.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(div_21.data_ptr()), c_void_p(div_22.data_ptr()), c_void_p(unsqueeze_47.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf183.data_ptr()))
    del div_21
    del primals_40
    del unsqueeze_47
    buf185 = reinterpret_tensor(buf173, (128, 48, 196), (9408, 196, 1), 0); del buf173  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_260, reinterpret_tensor(buf184, (128, 196, 196), (38416, 196, 1), 0), out=buf185)
    del permute_260
    buf186 = reinterpret_tensor(buf142, (128, 196, 48), (9408, 48, 1), 0); del buf142  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf184, (128, 196, 196), (38416, 196, 1), 0), permute_261, out=buf186)
    del permute_261
    buf188 = reinterpret_tensor(buf184, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf184  # reuse
    cpp_fused_clone_34(c_void_p(buf180.data_ptr()), c_void_p(div_22.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf188.data_ptr()))
    del div_22
    buf189 = empty((16, 3), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf188, (16, 307328), (1, 16), 0), view_8, out=buf189)
    buf190 = buf147; del buf147  # reuse
    cpp_fused_clone_35(c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf190.data_ptr()))
    buf191 = empty((1536, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf190, (1536, 1568), (1, 1536), 0), view_187, out=buf191)
    del view_187
    buf192 = reinterpret_tensor(buf186, (1568, 768), (768, 1), 0); del buf186  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf190, (1568, 1536), (1536, 1), 0), permute_270, out=buf192)
    del permute_270
    buf193 = buf163; del buf163  # reuse
    buf194 = buf162; del buf162  # reuse
    buf195 = empty((768, ), device='cpu', dtype=torch.float32)
    buf196 = empty((768, ), device='cpu', dtype=torch.float32)
    buf197 = buf166; del buf166  # reuse
    cpp_fused_add_native_layer_norm_backward_36(c_void_p(buf197.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(mul_70.data_ptr()), c_void_p(div_51.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()))
    del div_51
    del mul_70
    del primals_38
    buf198 = reinterpret_tensor(buf158, (1568, 3072), (3072, 1), 0); del buf158  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf197, (1568, 768), (768, 1), 0), permute_272, out=buf198)
    del permute_272
    buf199 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf197, (768, 1568), (1, 768), 0), view_181, out=buf199)
    del view_181
    buf200 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf201 = reinterpret_tensor(buf198, (8, 196, 3072), (602112, 3072, 1), 0); del buf198  # reuse
    cpp_fused_gelu_gelu_backward_sum_37(c_void_p(buf201.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(addmm_19.data_ptr()), c_void_p(buf200.data_ptr()))
    del addmm_19
    buf202 = buf192; del buf192  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf201, (1568, 3072), (3072, 1), 0), permute_276, out=buf202)
    del permute_276
    buf203 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf201, (3072, 1568), (1, 3072), 0), view_179, out=buf203)
    del view_179
    buf204 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf205 = buf194; del buf194  # reuse
    buf206 = buf193; del buf193  # reuse
    buf207 = empty((768, ), device='cpu', dtype=torch.float32)
    buf208 = empty((768, ), device='cpu', dtype=torch.float32)
    buf209 = buf197; del buf197  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_38(c_void_p(buf209.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(mul_65.data_ptr()), c_void_p(div_52.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()))
    del div_52
    del mul_65
    del primals_36
    buf210 = buf202; del buf202  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf209, (1568, 768), (768, 1), 0), permute_280, out=buf210)
    del permute_280
    buf211 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf209, (768, 1568), (1, 768), 0), view_177, out=buf211)
    del view_177
    buf212 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf213 = reinterpret_tensor(buf175, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf175  # reuse
    cpp_fused_clone_sum_39(c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()))
    buf214 = reinterpret_tensor(buf210, (128, 196, 48), (9408, 48, 1), 0); del buf210  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_285, reinterpret_tensor(buf213, (128, 196, 48), (9408, 48, 1), 0), out=buf214)
    del permute_285
    buf215 = reinterpret_tensor(buf188, (128, 196, 196), (38416, 196, 1), 0); del buf188  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf213, (128, 196, 48), (9408, 48, 1), 0), permute_286, out=buf215)
    del permute_286
    buf216 = reinterpret_tensor(buf213, (1568, 768), (768, 1), 0); del buf213  # reuse
    cpp_fused_view_40(c_void_p(buf214.data_ptr()), c_void_p(buf216.data_ptr()))
    buf217 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf216, (768, 1568), (1, 768), 0), view_161, out=buf217)
    buf218 = reinterpret_tensor(buf214, (1568, 768), (768, 1), 0); del buf214  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf216, permute_290, out=buf218)
    del permute_290
    buf219 = reinterpret_tensor(buf181, (8, 16, 196, 1), (3136, 196, 1, 1), 0); del buf181  # reuse
    buf223 = buf180; del buf180  # reuse
    buf225 = reinterpret_tensor(buf172, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf172  # reuse
    buf224 = buf183; del buf183  # reuse
    buf220 = buf178; del buf178  # reuse
    buf221 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf230 = empty((1, 1, 1, 16), device='cpu', dtype=torch.float32)
    buf222 = reinterpret_tensor(buf220, (16, ), (1, ), 0); del buf220  # reuse
    buf226 = reinterpret_tensor(buf176, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf176  # reuse
    buf227 = buf225; del buf225  # reuse
    cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_41(c_void_p(buf222.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(div_18.data_ptr()), c_void_p(div_19.data_ptr()), c_void_p(unsqueeze_41.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf226.data_ptr()))
    del div_18
    del primals_35
    del unsqueeze_41
    buf228 = reinterpret_tensor(buf216, (128, 48, 196), (9408, 196, 1), 0); del buf216  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_292, reinterpret_tensor(buf227, (128, 196, 196), (38416, 196, 1), 0), out=buf228)
    del permute_292
    buf229 = reinterpret_tensor(buf185, (128, 196, 48), (9408, 48, 1), 0); del buf185  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf227, (128, 196, 196), (38416, 196, 1), 0), permute_293, out=buf229)
    del permute_293
    buf231 = reinterpret_tensor(buf227, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf227  # reuse
    cpp_fused_clone_42(c_void_p(buf223.data_ptr()), c_void_p(div_19.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf231.data_ptr()))
    del div_19
    buf232 = empty((16, 3), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf231, (16, 307328), (1, 16), 0), view_8, out=buf232)
    buf233 = buf190; del buf190  # reuse
    cpp_fused_clone_43(c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf233.data_ptr()))
    buf234 = empty((1536, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf233, (1536, 1568), (1, 1536), 0), view_161, out=buf234)
    del view_161
    buf235 = reinterpret_tensor(buf229, (1568, 768), (768, 1), 0); del buf229  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf233, (1568, 1536), (1536, 1), 0), permute_302, out=buf235)
    del permute_302
    buf236 = buf206; del buf206  # reuse
    buf237 = buf205; del buf205  # reuse
    buf238 = empty((768, ), device='cpu', dtype=torch.float32)
    buf239 = empty((768, ), device='cpu', dtype=torch.float32)
    buf240 = buf209; del buf209  # reuse
    cpp_fused_add_native_layer_norm_backward_44(c_void_p(buf240.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(mul_60.data_ptr()), c_void_p(div_56.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()))
    del div_56
    del mul_60
    del primals_33
    buf241 = reinterpret_tensor(buf201, (1568, 3072), (3072, 1), 0); del buf201  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf240, (1568, 768), (768, 1), 0), permute_304, out=buf241)
    del permute_304
    buf242 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf240, (768, 1568), (1, 768), 0), view_155, out=buf242)
    del view_155
    buf243 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf244 = reinterpret_tensor(buf241, (8, 196, 3072), (602112, 3072, 1), 0); del buf241  # reuse
    cpp_fused_gelu_gelu_backward_sum_45(c_void_p(buf244.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(addmm_16.data_ptr()), c_void_p(buf243.data_ptr()))
    del addmm_16
    buf245 = buf235; del buf235  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf244, (1568, 3072), (3072, 1), 0), permute_308, out=buf245)
    del permute_308
    buf246 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf244, (3072, 1568), (1, 3072), 0), view_153, out=buf246)
    del view_153
    buf247 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf248 = buf237; del buf237  # reuse
    buf249 = buf236; del buf236  # reuse
    buf250 = empty((768, ), device='cpu', dtype=torch.float32)
    buf251 = empty((768, ), device='cpu', dtype=torch.float32)
    buf252 = buf240; del buf240  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_46(c_void_p(buf252.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(mul_55.data_ptr()), c_void_p(div_57.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()))
    del div_57
    del mul_55
    del primals_31
    buf253 = buf245; del buf245  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf252, (1568, 768), (768, 1), 0), permute_312, out=buf253)
    del permute_312
    buf254 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf252, (768, 1568), (1, 768), 0), view_151, out=buf254)
    del view_151
    buf255 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf256 = reinterpret_tensor(buf218, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf218  # reuse
    cpp_fused_clone_sum_47(c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()))
    buf257 = reinterpret_tensor(buf253, (128, 196, 48), (9408, 48, 1), 0); del buf253  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_317, reinterpret_tensor(buf256, (128, 196, 48), (9408, 48, 1), 0), out=buf257)
    del permute_317
    buf258 = reinterpret_tensor(buf231, (128, 196, 196), (38416, 196, 1), 0); del buf231  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf256, (128, 196, 48), (9408, 48, 1), 0), permute_318, out=buf258)
    del permute_318
    buf259 = reinterpret_tensor(buf256, (1568, 768), (768, 1), 0); del buf256  # reuse
    cpp_fused_view_48(c_void_p(buf257.data_ptr()), c_void_p(buf259.data_ptr()))
    buf260 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf259, (768, 1568), (1, 768), 0), view_135, out=buf260)
    buf261 = reinterpret_tensor(buf257, (1568, 768), (768, 1), 0); del buf257  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf259, permute_322, out=buf261)
    del permute_322
    buf262 = reinterpret_tensor(buf224, (8, 16, 196, 1), (3136, 196, 1, 1), 0); del buf224  # reuse
    buf266 = buf223; del buf223  # reuse
    buf268 = reinterpret_tensor(buf215, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf215  # reuse
    buf267 = buf226; del buf226  # reuse
    buf263 = buf221; del buf221  # reuse
    buf264 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf273 = empty((1, 1, 1, 16), device='cpu', dtype=torch.float32)
    buf265 = reinterpret_tensor(buf263, (16, ), (1, ), 0); del buf263  # reuse
    buf269 = reinterpret_tensor(buf219, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf219  # reuse
    buf270 = buf268; del buf268  # reuse
    cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_49(c_void_p(buf265.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(div_15.data_ptr()), c_void_p(div_16.data_ptr()), c_void_p(unsqueeze_35.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf269.data_ptr()))
    del div_15
    del primals_30
    del unsqueeze_35
    buf271 = reinterpret_tensor(buf259, (128, 48, 196), (9408, 196, 1), 0); del buf259  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_324, reinterpret_tensor(buf270, (128, 196, 196), (38416, 196, 1), 0), out=buf271)
    del permute_324
    buf272 = reinterpret_tensor(buf228, (128, 196, 48), (9408, 48, 1), 0); del buf228  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf270, (128, 196, 196), (38416, 196, 1), 0), permute_325, out=buf272)
    del permute_325
    buf274 = reinterpret_tensor(buf270, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf270  # reuse
    cpp_fused_clone_50(c_void_p(buf266.data_ptr()), c_void_p(div_16.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf274.data_ptr()))
    del div_16
    buf275 = empty((16, 3), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf274, (16, 307328), (1, 16), 0), view_8, out=buf275)
    buf276 = buf233; del buf233  # reuse
    cpp_fused_clone_51(c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf276.data_ptr()))
    buf277 = empty((1536, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf276, (1536, 1568), (1, 1536), 0), view_135, out=buf277)
    del view_135
    buf278 = reinterpret_tensor(buf272, (1568, 768), (768, 1), 0); del buf272  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf276, (1568, 1536), (1536, 1), 0), permute_334, out=buf278)
    del permute_334
    buf279 = buf249; del buf249  # reuse
    buf280 = buf248; del buf248  # reuse
    buf281 = empty((768, ), device='cpu', dtype=torch.float32)
    buf282 = empty((768, ), device='cpu', dtype=torch.float32)
    buf283 = buf252; del buf252  # reuse
    cpp_fused_add_native_layer_norm_backward_52(c_void_p(buf283.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(mul_50.data_ptr()), c_void_p(div_61.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()))
    del div_61
    del mul_50
    del primals_28
    buf284 = reinterpret_tensor(buf244, (1568, 3072), (3072, 1), 0); del buf244  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf283, (1568, 768), (768, 1), 0), permute_336, out=buf284)
    del permute_336
    buf285 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf283, (768, 1568), (1, 768), 0), view_129, out=buf285)
    del view_129
    buf286 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf287 = reinterpret_tensor(buf284, (8, 196, 3072), (602112, 3072, 1), 0); del buf284  # reuse
    cpp_fused_gelu_gelu_backward_sum_53(c_void_p(buf287.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(addmm_13.data_ptr()), c_void_p(buf286.data_ptr()))
    del addmm_13
    buf288 = buf278; del buf278  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf287, (1568, 3072), (3072, 1), 0), permute_340, out=buf288)
    del permute_340
    buf289 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf287, (3072, 1568), (1, 3072), 0), view_127, out=buf289)
    del view_127
    buf290 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf291 = buf280; del buf280  # reuse
    buf292 = buf279; del buf279  # reuse
    buf293 = empty((768, ), device='cpu', dtype=torch.float32)
    buf294 = empty((768, ), device='cpu', dtype=torch.float32)
    buf295 = buf283; del buf283  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_54(c_void_p(buf295.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(mul_45.data_ptr()), c_void_p(div_62.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()))
    del div_62
    del mul_45
    del primals_26
    buf296 = buf288; del buf288  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf295, (1568, 768), (768, 1), 0), permute_344, out=buf296)
    del permute_344
    buf297 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf295, (768, 1568), (1, 768), 0), view_125, out=buf297)
    del view_125
    buf298 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf299 = reinterpret_tensor(buf261, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf261  # reuse
    cpp_fused_clone_sum_55(c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()))
    buf300 = reinterpret_tensor(buf296, (128, 196, 48), (9408, 48, 1), 0); del buf296  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_349, reinterpret_tensor(buf299, (128, 196, 48), (9408, 48, 1), 0), out=buf300)
    del permute_349
    buf301 = reinterpret_tensor(buf274, (128, 196, 196), (38416, 196, 1), 0); del buf274  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf299, (128, 196, 48), (9408, 48, 1), 0), permute_350, out=buf301)
    del permute_350
    buf302 = reinterpret_tensor(buf299, (1568, 768), (768, 1), 0); del buf299  # reuse
    cpp_fused_view_56(c_void_p(buf300.data_ptr()), c_void_p(buf302.data_ptr()))
    buf303 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf302, (768, 1568), (1, 768), 0), view_109, out=buf303)
    buf304 = reinterpret_tensor(buf300, (1568, 768), (768, 1), 0); del buf300  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf302, permute_354, out=buf304)
    del permute_354
    buf305 = reinterpret_tensor(buf267, (8, 16, 196, 1), (3136, 196, 1, 1), 0); del buf267  # reuse
    buf309 = buf266; del buf266  # reuse
    buf311 = reinterpret_tensor(buf258, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf258  # reuse
    buf310 = buf269; del buf269  # reuse
    buf306 = buf264; del buf264  # reuse
    buf307 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf316 = empty((1, 1, 1, 16), device='cpu', dtype=torch.float32)
    buf308 = reinterpret_tensor(buf306, (16, ), (1, ), 0); del buf306  # reuse
    buf312 = reinterpret_tensor(buf262, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf262  # reuse
    buf313 = buf311; del buf311  # reuse
    cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_57(c_void_p(buf308.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(div_12.data_ptr()), c_void_p(div_13.data_ptr()), c_void_p(unsqueeze_29.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf312.data_ptr()))
    del div_12
    del primals_25
    del unsqueeze_29
    buf314 = reinterpret_tensor(buf302, (128, 48, 196), (9408, 196, 1), 0); del buf302  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_356, reinterpret_tensor(buf313, (128, 196, 196), (38416, 196, 1), 0), out=buf314)
    del permute_356
    buf315 = reinterpret_tensor(buf271, (128, 196, 48), (9408, 48, 1), 0); del buf271  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf313, (128, 196, 196), (38416, 196, 1), 0), permute_357, out=buf315)
    del permute_357
    buf317 = reinterpret_tensor(buf313, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf313  # reuse
    cpp_fused_clone_58(c_void_p(buf309.data_ptr()), c_void_p(div_13.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf317.data_ptr()))
    del div_13
    buf318 = empty((16, 3), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf317, (16, 307328), (1, 16), 0), view_8, out=buf318)
    buf319 = buf276; del buf276  # reuse
    cpp_fused_clone_59(c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf319.data_ptr()))
    buf320 = empty((1536, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf319, (1536, 1568), (1, 1536), 0), view_109, out=buf320)
    del view_109
    buf321 = reinterpret_tensor(buf315, (1568, 768), (768, 1), 0); del buf315  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf319, (1568, 1536), (1536, 1), 0), permute_366, out=buf321)
    del permute_366
    buf322 = buf292; del buf292  # reuse
    buf323 = buf291; del buf291  # reuse
    buf324 = empty((768, ), device='cpu', dtype=torch.float32)
    buf325 = empty((768, ), device='cpu', dtype=torch.float32)
    buf326 = buf295; del buf295  # reuse
    cpp_fused_add_native_layer_norm_backward_60(c_void_p(buf326.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(mul_40.data_ptr()), c_void_p(div_66.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()))
    del div_66
    del mul_40
    del primals_23
    buf327 = reinterpret_tensor(buf287, (1568, 3072), (3072, 1), 0); del buf287  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf326, (1568, 768), (768, 1), 0), permute_368, out=buf327)
    del permute_368
    buf328 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf326, (768, 1568), (1, 768), 0), view_103, out=buf328)
    del view_103
    buf329 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf330 = reinterpret_tensor(buf327, (8, 196, 3072), (602112, 3072, 1), 0); del buf327  # reuse
    cpp_fused_gelu_gelu_backward_sum_61(c_void_p(buf330.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(addmm_10.data_ptr()), c_void_p(buf329.data_ptr()))
    del addmm_10
    buf331 = buf321; del buf321  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf330, (1568, 3072), (3072, 1), 0), permute_372, out=buf331)
    del permute_372
    buf332 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf330, (3072, 1568), (1, 3072), 0), view_101, out=buf332)
    del view_101
    buf333 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf334 = buf323; del buf323  # reuse
    buf335 = buf322; del buf322  # reuse
    buf336 = empty((768, ), device='cpu', dtype=torch.float32)
    buf337 = empty((768, ), device='cpu', dtype=torch.float32)
    buf338 = buf326; del buf326  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_62(c_void_p(buf338.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(mul_35.data_ptr()), c_void_p(div_67.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()))
    del div_67
    del mul_35
    del primals_21
    buf339 = buf331; del buf331  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf338, (1568, 768), (768, 1), 0), permute_376, out=buf339)
    del permute_376
    buf340 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf338, (768, 1568), (1, 768), 0), view_99, out=buf340)
    del view_99
    buf341 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf342 = reinterpret_tensor(buf304, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf304  # reuse
    cpp_fused_clone_sum_63(c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()))
    buf343 = reinterpret_tensor(buf339, (128, 196, 48), (9408, 48, 1), 0); del buf339  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_381, reinterpret_tensor(buf342, (128, 196, 48), (9408, 48, 1), 0), out=buf343)
    del permute_381
    buf344 = reinterpret_tensor(buf317, (128, 196, 196), (38416, 196, 1), 0); del buf317  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf342, (128, 196, 48), (9408, 48, 1), 0), permute_382, out=buf344)
    del permute_382
    buf345 = reinterpret_tensor(buf342, (1568, 768), (768, 1), 0); del buf342  # reuse
    cpp_fused_view_64(c_void_p(buf343.data_ptr()), c_void_p(buf345.data_ptr()))
    buf346 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf345, (768, 1568), (1, 768), 0), view_83, out=buf346)
    buf347 = reinterpret_tensor(buf343, (1568, 768), (768, 1), 0); del buf343  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf345, permute_386, out=buf347)
    del permute_386
    buf348 = reinterpret_tensor(buf310, (8, 16, 196, 1), (3136, 196, 1, 1), 0); del buf310  # reuse
    buf352 = buf309; del buf309  # reuse
    buf354 = reinterpret_tensor(buf301, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf301  # reuse
    buf353 = buf312; del buf312  # reuse
    buf349 = buf307; del buf307  # reuse
    buf350 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf359 = empty((1, 1, 1, 16), device='cpu', dtype=torch.float32)
    buf351 = reinterpret_tensor(buf349, (16, ), (1, ), 0); del buf349  # reuse
    buf355 = reinterpret_tensor(buf305, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf305  # reuse
    buf356 = buf354; del buf354  # reuse
    cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_65(c_void_p(buf351.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(div_9.data_ptr()), c_void_p(div_10.data_ptr()), c_void_p(unsqueeze_23.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf355.data_ptr()))
    del div_9
    del primals_20
    del unsqueeze_23
    buf357 = reinterpret_tensor(buf345, (128, 48, 196), (9408, 196, 1), 0); del buf345  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_388, reinterpret_tensor(buf356, (128, 196, 196), (38416, 196, 1), 0), out=buf357)
    del permute_388
    buf358 = reinterpret_tensor(buf314, (128, 196, 48), (9408, 48, 1), 0); del buf314  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf356, (128, 196, 196), (38416, 196, 1), 0), permute_389, out=buf358)
    del permute_389
    buf360 = reinterpret_tensor(buf356, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf356  # reuse
    cpp_fused_clone_66(c_void_p(buf352.data_ptr()), c_void_p(div_10.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf360.data_ptr()))
    del div_10
    buf361 = empty((16, 3), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf360, (16, 307328), (1, 16), 0), view_8, out=buf361)
    buf362 = buf319; del buf319  # reuse
    cpp_fused_clone_67(c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf362.data_ptr()))
    buf363 = empty((1536, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf362, (1536, 1568), (1, 1536), 0), view_83, out=buf363)
    del view_83
    buf364 = reinterpret_tensor(buf358, (1568, 768), (768, 1), 0); del buf358  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf362, (1568, 1536), (1536, 1), 0), permute_398, out=buf364)
    del permute_398
    buf365 = buf335; del buf335  # reuse
    buf366 = buf334; del buf334  # reuse
    buf367 = empty((768, ), device='cpu', dtype=torch.float32)
    buf368 = empty((768, ), device='cpu', dtype=torch.float32)
    buf369 = buf338; del buf338  # reuse
    cpp_fused_add_native_layer_norm_backward_68(c_void_p(buf369.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(mul_30.data_ptr()), c_void_p(div_71.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()))
    del div_71
    del mul_30
    del primals_18
    buf370 = reinterpret_tensor(buf330, (1568, 3072), (3072, 1), 0); del buf330  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf369, (1568, 768), (768, 1), 0), permute_400, out=buf370)
    del permute_400
    buf371 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf369, (768, 1568), (1, 768), 0), view_77, out=buf371)
    del view_77
    buf372 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf373 = reinterpret_tensor(buf370, (8, 196, 3072), (602112, 3072, 1), 0); del buf370  # reuse
    cpp_fused_gelu_gelu_backward_sum_69(c_void_p(buf373.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(addmm_7.data_ptr()), c_void_p(buf372.data_ptr()))
    del addmm_7
    buf374 = buf364; del buf364  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf373, (1568, 3072), (3072, 1), 0), permute_404, out=buf374)
    del permute_404
    buf375 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf373, (3072, 1568), (1, 3072), 0), view_75, out=buf375)
    del view_75
    buf376 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf377 = buf366; del buf366  # reuse
    buf378 = buf365; del buf365  # reuse
    buf379 = empty((768, ), device='cpu', dtype=torch.float32)
    buf380 = empty((768, ), device='cpu', dtype=torch.float32)
    buf381 = buf369; del buf369  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_70(c_void_p(buf381.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(mul_25.data_ptr()), c_void_p(div_72.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()))
    del div_72
    del mul_25
    del primals_16
    buf382 = buf374; del buf374  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf381, (1568, 768), (768, 1), 0), permute_408, out=buf382)
    del permute_408
    buf383 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf381, (768, 1568), (1, 768), 0), view_73, out=buf383)
    del view_73
    buf384 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf385 = reinterpret_tensor(buf347, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf347  # reuse
    cpp_fused_clone_sum_71(c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()))
    buf386 = reinterpret_tensor(buf382, (128, 196, 48), (9408, 48, 1), 0); del buf382  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_413, reinterpret_tensor(buf385, (128, 196, 48), (9408, 48, 1), 0), out=buf386)
    del permute_413
    buf387 = reinterpret_tensor(buf360, (128, 196, 196), (38416, 196, 1), 0); del buf360  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf385, (128, 196, 48), (9408, 48, 1), 0), permute_414, out=buf387)
    del permute_414
    buf388 = reinterpret_tensor(buf385, (1568, 768), (768, 1), 0); del buf385  # reuse
    cpp_fused_view_72(c_void_p(buf386.data_ptr()), c_void_p(buf388.data_ptr()))
    buf389 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf388, (768, 1568), (1, 768), 0), view_57, out=buf389)
    buf390 = reinterpret_tensor(buf386, (1568, 768), (768, 1), 0); del buf386  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf388, permute_418, out=buf390)
    del permute_418
    buf391 = reinterpret_tensor(buf353, (8, 16, 196, 1), (3136, 196, 1, 1), 0); del buf353  # reuse
    buf395 = buf352; del buf352  # reuse
    buf397 = reinterpret_tensor(buf344, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf344  # reuse
    buf396 = buf355; del buf355  # reuse
    buf392 = buf350; del buf350  # reuse
    buf393 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf402 = empty((1, 1, 1, 16), device='cpu', dtype=torch.float32)
    buf394 = reinterpret_tensor(buf392, (16, ), (1, ), 0); del buf392  # reuse
    buf398 = reinterpret_tensor(buf348, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf348  # reuse
    buf399 = buf397; del buf397  # reuse
    cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_73(c_void_p(buf394.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(div_6.data_ptr()), c_void_p(div_7.data_ptr()), c_void_p(unsqueeze_17.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf398.data_ptr()))
    del div_6
    del primals_15
    del unsqueeze_17
    buf400 = reinterpret_tensor(buf388, (128, 48, 196), (9408, 196, 1), 0); del buf388  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_420, reinterpret_tensor(buf399, (128, 196, 196), (38416, 196, 1), 0), out=buf400)
    del permute_420
    buf401 = reinterpret_tensor(buf357, (128, 196, 48), (9408, 48, 1), 0); del buf357  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf399, (128, 196, 196), (38416, 196, 1), 0), permute_421, out=buf401)
    del permute_421
    buf403 = reinterpret_tensor(buf399, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf399  # reuse
    cpp_fused_clone_74(c_void_p(buf395.data_ptr()), c_void_p(div_7.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf403.data_ptr()))
    del div_7
    buf404 = empty((16, 3), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf403, (16, 307328), (1, 16), 0), view_8, out=buf404)
    buf405 = buf362; del buf362  # reuse
    cpp_fused_clone_75(c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf405.data_ptr()))
    buf406 = empty((1536, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf405, (1536, 1568), (1, 1536), 0), view_57, out=buf406)
    del view_57
    buf407 = reinterpret_tensor(buf401, (1568, 768), (768, 1), 0); del buf401  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf405, (1568, 1536), (1536, 1), 0), permute_430, out=buf407)
    del permute_430
    buf408 = buf378; del buf378  # reuse
    buf409 = buf377; del buf377  # reuse
    buf410 = empty((768, ), device='cpu', dtype=torch.float32)
    buf411 = empty((768, ), device='cpu', dtype=torch.float32)
    buf412 = buf381; del buf381  # reuse
    cpp_fused_add_native_layer_norm_backward_76(c_void_p(buf412.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(mul_20.data_ptr()), c_void_p(div_76.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf411.data_ptr()))
    del div_76
    del mul_20
    del primals_13
    buf413 = reinterpret_tensor(buf373, (1568, 3072), (3072, 1), 0); del buf373  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf412, (1568, 768), (768, 1), 0), permute_432, out=buf413)
    del permute_432
    buf414 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf412, (768, 1568), (1, 768), 0), view_51, out=buf414)
    del view_51
    buf415 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf416 = reinterpret_tensor(buf413, (8, 196, 3072), (602112, 3072, 1), 0); del buf413  # reuse
    cpp_fused_gelu_gelu_backward_sum_77(c_void_p(buf416.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(addmm_4.data_ptr()), c_void_p(buf415.data_ptr()))
    del addmm_4
    buf417 = buf407; del buf407  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf416, (1568, 3072), (3072, 1), 0), permute_436, out=buf417)
    del permute_436
    buf418 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf416, (3072, 1568), (1, 3072), 0), view_49, out=buf418)
    del view_49
    buf419 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf420 = buf409; del buf409  # reuse
    buf421 = buf408; del buf408  # reuse
    buf422 = empty((768, ), device='cpu', dtype=torch.float32)
    buf423 = empty((768, ), device='cpu', dtype=torch.float32)
    buf424 = buf412; del buf412  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_78(c_void_p(buf424.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(mul_15.data_ptr()), c_void_p(div_77.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf423.data_ptr()))
    del div_77
    del mul_15
    del primals_11
    buf425 = buf417; del buf417  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf424, (1568, 768), (768, 1), 0), permute_440, out=buf425)
    del permute_440
    buf426 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf424, (768, 1568), (1, 768), 0), view_47, out=buf426)
    del view_47
    buf427 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf428 = reinterpret_tensor(buf390, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf390  # reuse
    cpp_fused_clone_sum_79(c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()))
    buf429 = reinterpret_tensor(buf425, (128, 196, 48), (9408, 48, 1), 0); del buf425  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_445, reinterpret_tensor(buf428, (128, 196, 48), (9408, 48, 1), 0), out=buf429)
    del permute_445
    buf430 = reinterpret_tensor(buf403, (128, 196, 196), (38416, 196, 1), 0); del buf403  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf428, (128, 196, 48), (9408, 48, 1), 0), permute_446, out=buf430)
    del permute_446
    buf431 = reinterpret_tensor(buf428, (1568, 768), (768, 1), 0); del buf428  # reuse
    cpp_fused_view_80(c_void_p(buf429.data_ptr()), c_void_p(buf431.data_ptr()))
    buf432 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf431, (768, 1568), (1, 768), 0), view_31, out=buf432)
    buf433 = reinterpret_tensor(buf429, (1568, 768), (768, 1), 0); del buf429  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf431, permute_450, out=buf433)
    del permute_450
    buf434 = reinterpret_tensor(buf396, (8, 16, 196, 1), (3136, 196, 1, 1), 0); del buf396  # reuse
    buf438 = buf395; del buf395  # reuse
    buf440 = reinterpret_tensor(buf387, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf387  # reuse
    buf439 = buf398; del buf398  # reuse
    buf435 = buf393; del buf393  # reuse
    buf436 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf445 = empty((1, 1, 1, 16), device='cpu', dtype=torch.float32)
    buf437 = reinterpret_tensor(buf435, (16, ), (1, ), 0); del buf435  # reuse
    buf441 = reinterpret_tensor(buf391, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf391  # reuse
    buf442 = buf440; del buf440  # reuse
    cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_81(c_void_p(buf437.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(div_3.data_ptr()), c_void_p(div_4.data_ptr()), c_void_p(unsqueeze_11.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf441.data_ptr()))
    del div_3
    del primals_10
    del unsqueeze_11
    buf443 = reinterpret_tensor(buf431, (128, 48, 196), (9408, 196, 1), 0); del buf431  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_452, reinterpret_tensor(buf442, (128, 196, 196), (38416, 196, 1), 0), out=buf443)
    del permute_452
    buf444 = reinterpret_tensor(buf400, (128, 196, 48), (9408, 48, 1), 0); del buf400  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf442, (128, 196, 196), (38416, 196, 1), 0), permute_453, out=buf444)
    del permute_453
    buf446 = reinterpret_tensor(buf442, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf442  # reuse
    cpp_fused_clone_82(c_void_p(buf438.data_ptr()), c_void_p(div_4.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf446.data_ptr()))
    del div_4
    buf447 = empty((16, 3), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf446, (16, 307328), (1, 16), 0), view_8, out=buf447)
    buf448 = buf405; del buf405  # reuse
    cpp_fused_clone_83(c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf448.data_ptr()))
    buf449 = empty((1536, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf448, (1536, 1568), (1, 1536), 0), view_31, out=buf449)
    del view_31
    buf450 = reinterpret_tensor(buf444, (1568, 768), (768, 1), 0); del buf444  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf448, (1568, 1536), (1536, 1), 0), permute_462, out=buf450)
    del permute_462
    buf451 = buf421; del buf421  # reuse
    buf452 = buf420; del buf420  # reuse
    buf453 = empty((768, ), device='cpu', dtype=torch.float32)
    buf454 = empty((768, ), device='cpu', dtype=torch.float32)
    buf455 = buf424; del buf424  # reuse
    cpp_fused_add_native_layer_norm_backward_84(c_void_p(buf455.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(mul_10.data_ptr()), c_void_p(div_81.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf454.data_ptr()))
    del div_81
    del mul_10
    del primals_8
    buf456 = reinterpret_tensor(buf416, (1568, 3072), (3072, 1), 0); del buf416  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf455, (1568, 768), (768, 1), 0), permute_464, out=buf456)
    del permute_464
    buf457 = empty((768, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf455, (768, 1568), (1, 768), 0), view_25, out=buf457)
    del view_25
    buf458 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf459 = reinterpret_tensor(buf456, (8, 196, 3072), (602112, 3072, 1), 0); del buf456  # reuse
    cpp_fused_gelu_gelu_backward_sum_85(c_void_p(buf459.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(addmm_1.data_ptr()), c_void_p(buf458.data_ptr()))
    del addmm_1
    buf460 = buf450; del buf450  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf459, (1568, 3072), (3072, 1), 0), permute_468, out=buf460)
    del permute_468
    buf461 = empty((3072, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf459, (3072, 1568), (1, 3072), 0), view_23, out=buf461)
    del view_23
    buf462 = empty((1, 3072), device='cpu', dtype=torch.float32)
    buf463 = buf452; del buf452  # reuse
    buf464 = buf451; del buf451  # reuse
    buf465 = empty((768, ), device='cpu', dtype=torch.float32)
    buf466 = empty((768, ), device='cpu', dtype=torch.float32)
    buf467 = buf455; del buf455  # reuse
    cpp_fused_add_native_layer_norm_backward_sum_86(c_void_p(buf467.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(mul_5.data_ptr()), c_void_p(div_82.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()))
    del buf459
    del div_82
    del mul_5
    del primals_6
    buf468 = buf460; del buf460  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf467, (1568, 768), (768, 1), 0), permute_472, out=buf468)
    del permute_472
    buf469 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf467, (768, 1568), (1, 768), 0), view_21, out=buf469)
    del view_21
    buf470 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf471 = reinterpret_tensor(buf433, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf433  # reuse
    cpp_fused_clone_sum_87(c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()))
    buf472 = reinterpret_tensor(buf468, (128, 196, 48), (9408, 48, 1), 0); del buf468  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_477, reinterpret_tensor(buf471, (128, 196, 48), (9408, 48, 1), 0), out=buf472)
    del permute_477
    buf473 = reinterpret_tensor(buf446, (128, 196, 196), (38416, 196, 1), 0); del buf446  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf471, (128, 196, 48), (9408, 48, 1), 0), permute_478, out=buf473)
    del permute_478
    buf474 = reinterpret_tensor(buf471, (1568, 768), (768, 1), 0); del buf471  # reuse
    cpp_fused_view_88(c_void_p(buf472.data_ptr()), c_void_p(buf474.data_ptr()))
    buf475 = empty((768, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf474, (768, 1568), (1, 768), 0), view_5, out=buf475)
    buf476 = reinterpret_tensor(buf472, (1568, 768), (768, 1), 0); del buf472  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf474, permute_482, out=buf476)
    del permute_482
    buf477 = reinterpret_tensor(buf439, (8, 16, 196, 1), (3136, 196, 1, 1), 0); del buf439  # reuse
    buf481 = buf438; del buf438  # reuse
    buf483 = reinterpret_tensor(buf430, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf430  # reuse
    buf482 = buf441; del buf441  # reuse
    buf478 = buf436; del buf436  # reuse
    buf479 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf488 = empty((1, 1, 1, 16), device='cpu', dtype=torch.float32)
    buf480 = reinterpret_tensor(buf478, (16, ), (1, ), 0); del buf478  # reuse
    buf484 = reinterpret_tensor(buf434, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf434  # reuse
    buf485 = buf483; del buf483  # reuse
    cpp_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sigmoid_backward_sum_view_89(c_void_p(buf480.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(div.data_ptr()), c_void_p(div_1.data_ptr()), c_void_p(unsqueeze_5.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf484.data_ptr()))
    del buf473
    del buf477
    del buf479
    del buf484
    del div
    del primals_5
    del unsqueeze_5
    buf486 = reinterpret_tensor(buf474, (128, 48, 196), (9408, 196, 1), 0); del buf474  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(permute_484, reinterpret_tensor(buf485, (128, 196, 196), (38416, 196, 1), 0), out=buf486)
    del permute_484
    buf487 = reinterpret_tensor(buf443, (128, 196, 48), (9408, 48, 1), 0); del buf443  # reuse
    # Source Nodes: [], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf485, (128, 196, 196), (38416, 196, 1), 0), permute_485, out=buf487)
    del permute_485
    buf489 = reinterpret_tensor(buf485, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf485  # reuse
    cpp_fused_clone_90(c_void_p(buf481.data_ptr()), c_void_p(div_1.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf489.data_ptr()))
    del buf481
    del buf482
    del div_1
    buf490 = empty((16, 3), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf489, (16, 307328), (1, 16), 0), view_8, out=buf490)
    del buf489
    del view_8
    buf491 = buf448; del buf448  # reuse
    cpp_fused_clone_91(c_void_p(buf486.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf491.data_ptr()))
    del buf486
    buf492 = empty((1536, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf491, (1536, 1568), (1, 1536), 0), view_5, out=buf492)
    del view_5
    buf493 = reinterpret_tensor(buf487, (1568, 768), (768, 1), 0); del buf487  # reuse
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf491, (1568, 1536), (1536, 1), 0), permute_494, out=buf493)
    del buf491
    del permute_494
    buf494 = buf464; del buf464  # reuse
    buf495 = buf463; del buf463  # reuse
    buf496 = empty((768, ), device='cpu', dtype=torch.float32)
    buf497 = empty((768, ), device='cpu', dtype=torch.float32)
    buf498 = buf467; del buf467  # reuse
    buf499 = empty((1, 1, 768), device='cpu', dtype=torch.float32)
    buf500 = empty((1, 196, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_backward_sum_92(c_void_p(buf498.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(mul.data_ptr()), c_void_p(div_86.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()))
    del buf476
    del buf493
    del buf494
    del buf495
    del buf67
    del div_86
    del mul
    del primals_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf501 = aten.convolution_backward(reinterpret_tensor(buf498, (8, 768, 14, 14), (150528, 1, 10752, 768), 0), primals_181, primals_63, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf498
    del primals_181
    del primals_63
    buf502 = buf501[1]
    buf503 = buf501[2]
    return (buf500, buf499, buf496, buf497, buf480, buf465, buf466, buf453, buf454, buf437, buf422, buf423, buf410, buf411, buf394, buf379, buf380, buf367, buf368, buf351, buf336, buf337, buf324, buf325, buf308, buf293, buf294, buf281, buf282, buf265, buf250, buf251, buf238, buf239, buf222, buf207, buf208, buf195, buf196, buf179, buf164, buf165, buf152, buf153, buf136, buf121, buf122, buf109, buf110, buf93, buf78, buf79, buf65, buf66, buf47, buf48, buf35, buf36, buf17, buf18, buf6, buf7, buf502, buf503, reinterpret_tensor(buf492, (1536, 768), (768, 1), 0), reinterpret_tensor(buf490, (16, 3), (3, 1), 0), reinterpret_tensor(buf488, (16, ), (1, ), 0), reinterpret_tensor(buf475, (768, 768), (768, 1), 0), reinterpret_tensor(buf469, (768, 768), (768, 1), 0), reinterpret_tensor(buf470, (768, ), (1, ), 0), reinterpret_tensor(buf461, (3072, 768), (768, 1), 0), reinterpret_tensor(buf462, (3072, ), (1, ), 0), reinterpret_tensor(buf457, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf458, (768, ), (1, ), 0), reinterpret_tensor(buf449, (1536, 768), (768, 1), 0), reinterpret_tensor(buf447, (16, 3), (3, 1), 0), reinterpret_tensor(buf445, (16, ), (1, ), 0), reinterpret_tensor(buf432, (768, 768), (768, 1), 0), reinterpret_tensor(buf426, (768, 768), (768, 1), 0), reinterpret_tensor(buf427, (768, ), (1, ), 0), reinterpret_tensor(buf418, (3072, 768), (768, 1), 0), reinterpret_tensor(buf419, (3072, ), (1, ), 0), reinterpret_tensor(buf414, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf415, (768, ), (1, ), 0), reinterpret_tensor(buf406, (1536, 768), (768, 1), 0), reinterpret_tensor(buf404, (16, 3), (3, 1), 0), reinterpret_tensor(buf402, (16, ), (1, ), 0), reinterpret_tensor(buf389, (768, 768), (768, 1), 0), reinterpret_tensor(buf383, (768, 768), (768, 1), 0), reinterpret_tensor(buf384, (768, ), (1, ), 0), reinterpret_tensor(buf375, (3072, 768), (768, 1), 0), reinterpret_tensor(buf376, (3072, ), (1, ), 0), reinterpret_tensor(buf371, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf372, (768, ), (1, ), 0), reinterpret_tensor(buf363, (1536, 768), (768, 1), 0), reinterpret_tensor(buf361, (16, 3), (3, 1), 0), reinterpret_tensor(buf359, (16, ), (1, ), 0), reinterpret_tensor(buf346, (768, 768), (768, 1), 0), reinterpret_tensor(buf340, (768, 768), (768, 1), 0), reinterpret_tensor(buf341, (768, ), (1, ), 0), reinterpret_tensor(buf332, (3072, 768), (768, 1), 0), reinterpret_tensor(buf333, (3072, ), (1, ), 0), reinterpret_tensor(buf328, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf329, (768, ), (1, ), 0), reinterpret_tensor(buf320, (1536, 768), (768, 1), 0), reinterpret_tensor(buf318, (16, 3), (3, 1), 0), reinterpret_tensor(buf316, (16, ), (1, ), 0), reinterpret_tensor(buf303, (768, 768), (768, 1), 0), reinterpret_tensor(buf297, (768, 768), (768, 1), 0), reinterpret_tensor(buf298, (768, ), (1, ), 0), reinterpret_tensor(buf289, (3072, 768), (768, 1), 0), reinterpret_tensor(buf290, (3072, ), (1, ), 0), reinterpret_tensor(buf285, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf286, (768, ), (1, ), 0), reinterpret_tensor(buf277, (1536, 768), (768, 1), 0), reinterpret_tensor(buf275, (16, 3), (3, 1), 0), reinterpret_tensor(buf273, (16, ), (1, ), 0), reinterpret_tensor(buf260, (768, 768), (768, 1), 0), reinterpret_tensor(buf254, (768, 768), (768, 1), 0), reinterpret_tensor(buf255, (768, ), (1, ), 0), reinterpret_tensor(buf246, (3072, 768), (768, 1), 0), reinterpret_tensor(buf247, (3072, ), (1, ), 0), reinterpret_tensor(buf242, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf243, (768, ), (1, ), 0), reinterpret_tensor(buf234, (1536, 768), (768, 1), 0), reinterpret_tensor(buf232, (16, 3), (3, 1), 0), reinterpret_tensor(buf230, (16, ), (1, ), 0), reinterpret_tensor(buf217, (768, 768), (768, 1), 0), reinterpret_tensor(buf211, (768, 768), (768, 1), 0), reinterpret_tensor(buf212, (768, ), (1, ), 0), reinterpret_tensor(buf203, (3072, 768), (768, 1), 0), reinterpret_tensor(buf204, (3072, ), (1, ), 0), reinterpret_tensor(buf199, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf200, (768, ), (1, ), 0), reinterpret_tensor(buf191, (1536, 768), (768, 1), 0), reinterpret_tensor(buf189, (16, 3), (3, 1), 0), reinterpret_tensor(buf187, (16, ), (1, ), 0), reinterpret_tensor(buf174, (768, 768), (768, 1), 0), reinterpret_tensor(buf168, (768, 768), (768, 1), 0), reinterpret_tensor(buf169, (768, ), (1, ), 0), reinterpret_tensor(buf160, (3072, 768), (768, 1), 0), reinterpret_tensor(buf161, (3072, ), (1, ), 0), reinterpret_tensor(buf156, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf157, (768, ), (1, ), 0), reinterpret_tensor(buf148, (1536, 768), (768, 1), 0), reinterpret_tensor(buf146, (16, 3), (3, 1), 0), reinterpret_tensor(buf144, (16, ), (1, ), 0), reinterpret_tensor(buf131, (768, 768), (768, 1), 0), reinterpret_tensor(buf125, (768, 768), (768, 1), 0), reinterpret_tensor(buf126, (768, ), (1, ), 0), reinterpret_tensor(buf117, (3072, 768), (768, 1), 0), reinterpret_tensor(buf118, (3072, ), (1, ), 0), reinterpret_tensor(buf113, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf114, (768, ), (1, ), 0), reinterpret_tensor(buf105, (1536, 768), (768, 1), 0), reinterpret_tensor(buf103, (16, 3), (3, 1), 0), reinterpret_tensor(buf101, (16, ), (1, ), 0), reinterpret_tensor(buf88, (768, 768), (768, 1), 0), reinterpret_tensor(buf82, (768, 768), (768, 1), 0), reinterpret_tensor(buf83, (768, ), (1, ), 0), reinterpret_tensor(buf74, (3072, 768), (768, 1), 0), reinterpret_tensor(buf75, (3072, ), (1, ), 0), reinterpret_tensor(buf70, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf71, (768, ), (1, ), 0), reinterpret_tensor(buf61, (2304, 768), (768, 1), 0), reinterpret_tensor(buf51, (768, 768), (768, 1), 0), reinterpret_tensor(buf52, (768, ), (1, ), 0), reinterpret_tensor(buf43, (3072, 768), (768, 1), 0), reinterpret_tensor(buf44, (3072, ), (1, ), 0), reinterpret_tensor(buf39, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf40, (768, ), (1, ), 0), reinterpret_tensor(buf31, (2304, 768), (768, 1), 0), reinterpret_tensor(buf21, (768, 768), (768, 1), 0), reinterpret_tensor(buf22, (768, ), (1, ), 0), reinterpret_tensor(buf13, (3072, 768), (768, 1), 0), reinterpret_tensor(buf14, (3072, ), (1, ), 0), reinterpret_tensor(buf9, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf10, (768, ), (1, ), 0), reinterpret_tensor(buf1, (1000, 768), (768, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((768, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    mul = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_5 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_8 = rand_strided((307328, 3), (3, 1), device='cpu', dtype=torch.float32)
    div = rand_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    div_1 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cpu', dtype=torch.float32)
    unsqueeze_5 = rand_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    view_21 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_5 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_23 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_1 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_25 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_10 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_31 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_3 = rand_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    div_4 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cpu', dtype=torch.float32)
    unsqueeze_11 = rand_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    view_47 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_15 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_49 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_4 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_51 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_20 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_57 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_6 = rand_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    div_7 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cpu', dtype=torch.float32)
    unsqueeze_17 = rand_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    view_73 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_25 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_75 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_7 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_77 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_30 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_83 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_9 = rand_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    div_10 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cpu', dtype=torch.float32)
    unsqueeze_23 = rand_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    view_99 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_35 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_101 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_10 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_103 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_40 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_109 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_12 = rand_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    div_13 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cpu', dtype=torch.float32)
    unsqueeze_29 = rand_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    view_125 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_45 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_127 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_13 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_129 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_50 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_135 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_15 = rand_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    div_16 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cpu', dtype=torch.float32)
    unsqueeze_35 = rand_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    view_151 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_55 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_153 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_16 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_155 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_60 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_161 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_18 = rand_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    div_19 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cpu', dtype=torch.float32)
    unsqueeze_41 = rand_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    view_177 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_65 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_179 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_19 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_181 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_70 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_187 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_21 = rand_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    div_22 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cpu', dtype=torch.float32)
    unsqueeze_47 = rand_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    view_203 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_75 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_205 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_22 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_207 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_80 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_213 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    div_25 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cpu', dtype=torch.float32)
    unsqueeze_53 = rand_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    view_229 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_85 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_231 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_25 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_233 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_90 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_239 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_27 = rand_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    div_28 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cpu', dtype=torch.float32)
    unsqueeze_59 = rand_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    view_255 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_95 = rand_strided((8, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    view_257 = rand_strided((1568, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_28 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_259 = rand_strided((1568, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    cat = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    getitem_41 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    rsqrt_20 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    view_261 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_271 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_103 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_273 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_31 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_275 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_108 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_277 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    view_287 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    mul_111 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    view_289 = rand_strided((1576, 768), (768, 1), device='cpu', dtype=torch.float32)
    addmm_34 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    view_291 = rand_strided((1576, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    mul_116 = rand_strided((8, 197, 768), (151296, 768, 1), device='cpu', dtype=torch.float32)
    clone_167 = rand_strided((8, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_126 = rand_strided((1000, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_32 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_130 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_134 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_33 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_138 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_143 = rand_strided((128, 197, 197), (38809, 1, 197), device='cpu', dtype=torch.float32)
    permute_144 = rand_strided((128, 48, 197), (9456, 1, 48), device='cpu', dtype=torch.float32)
    alias_42 = rand_strided((8, 16, 197, 197), (620944, 1, 3152, 16), device='cpu', dtype=torch.float32)
    permute_145 = rand_strided((128, 48, 197), (9456, 1, 48), device='cpu', dtype=torch.float32)
    permute_146 = rand_strided((128, 197, 48), (9456, 1, 197), device='cpu', dtype=torch.float32)
    permute_151 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_34 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_153 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_157 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_35 = rand_strided((8, 197, 1), (197, 1, 1), device='cpu', dtype=torch.float32)
    permute_161 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_166 = rand_strided((128, 197, 197), (38809, 1, 197), device='cpu', dtype=torch.float32)
    permute_167 = rand_strided((128, 48, 197), (9456, 1, 48), device='cpu', dtype=torch.float32)
    alias_43 = rand_strided((8, 16, 197, 197), (620944, 1, 3152, 16), device='cpu', dtype=torch.float32)
    permute_168 = rand_strided((128, 48, 197), (9456, 1, 48), device='cpu', dtype=torch.float32)
    permute_169 = rand_strided((128, 197, 48), (9456, 1, 197), device='cpu', dtype=torch.float32)
    permute_174 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_176 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_180 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_37 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_184 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_189 = rand_strided((128, 196, 196), (38416, 1, 196), device='cpu', dtype=torch.float32)
    permute_190 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_194 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_196 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_197 = rand_strided((128, 196, 48), (9408, 1, 196), device='cpu', dtype=torch.float32)
    permute_206 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_41 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_208 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_212 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_42 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_216 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_221 = rand_strided((128, 196, 196), (38416, 1, 196), device='cpu', dtype=torch.float32)
    permute_222 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_226 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_228 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_229 = rand_strided((128, 196, 48), (9408, 1, 196), device='cpu', dtype=torch.float32)
    permute_238 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_46 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_240 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_244 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_47 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_248 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_253 = rand_strided((128, 196, 196), (38416, 1, 196), device='cpu', dtype=torch.float32)
    permute_254 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_258 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_260 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_261 = rand_strided((128, 196, 48), (9408, 1, 196), device='cpu', dtype=torch.float32)
    permute_270 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_51 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_272 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_276 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_52 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_280 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_285 = rand_strided((128, 196, 196), (38416, 1, 196), device='cpu', dtype=torch.float32)
    permute_286 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_290 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_292 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_293 = rand_strided((128, 196, 48), (9408, 1, 196), device='cpu', dtype=torch.float32)
    permute_302 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_56 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_304 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_308 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_57 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_312 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_317 = rand_strided((128, 196, 196), (38416, 1, 196), device='cpu', dtype=torch.float32)
    permute_318 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_322 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_324 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_325 = rand_strided((128, 196, 48), (9408, 1, 196), device='cpu', dtype=torch.float32)
    permute_334 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_61 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_336 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_340 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_62 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_344 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_349 = rand_strided((128, 196, 196), (38416, 1, 196), device='cpu', dtype=torch.float32)
    permute_350 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_354 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_356 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_357 = rand_strided((128, 196, 48), (9408, 1, 196), device='cpu', dtype=torch.float32)
    permute_366 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_66 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_368 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_372 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_67 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_376 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_381 = rand_strided((128, 196, 196), (38416, 1, 196), device='cpu', dtype=torch.float32)
    permute_382 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_386 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_388 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_389 = rand_strided((128, 196, 48), (9408, 1, 196), device='cpu', dtype=torch.float32)
    permute_398 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_71 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_400 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_404 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_72 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_408 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_413 = rand_strided((128, 196, 196), (38416, 1, 196), device='cpu', dtype=torch.float32)
    permute_414 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_418 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_420 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_421 = rand_strided((128, 196, 48), (9408, 1, 196), device='cpu', dtype=torch.float32)
    permute_430 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_76 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_432 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_436 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_77 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_440 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_445 = rand_strided((128, 196, 196), (38416, 1, 196), device='cpu', dtype=torch.float32)
    permute_446 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_450 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_452 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_453 = rand_strided((128, 196, 48), (9408, 1, 196), device='cpu', dtype=torch.float32)
    permute_462 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_81 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_464 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    permute_468 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_82 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    permute_472 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_477 = rand_strided((128, 196, 196), (38416, 1, 196), device='cpu', dtype=torch.float32)
    permute_478 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_482 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_484 = rand_strided((128, 48, 196), (9408, 1, 48), device='cpu', dtype=torch.float32)
    permute_485 = rand_strided((128, 196, 48), (9408, 1, 196), device='cpu', dtype=torch.float32)
    permute_494 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    div_86 = rand_strided((8, 196, 1), (196, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_3, primals_5, primals_6, primals_8, primals_10, primals_11, primals_13, primals_15, primals_16, primals_18, primals_20, primals_21, primals_23, primals_25, primals_26, primals_28, primals_30, primals_31, primals_33, primals_35, primals_36, primals_38, primals_40, primals_41, primals_43, primals_45, primals_46, primals_48, primals_50, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_181, mul, view_5, view_8, div, div_1, unsqueeze_5, view_21, mul_5, view_23, addmm_1, view_25, mul_10, view_31, div_3, div_4, unsqueeze_11, view_47, mul_15, view_49, addmm_4, view_51, mul_20, view_57, div_6, div_7, unsqueeze_17, view_73, mul_25, view_75, addmm_7, view_77, mul_30, view_83, div_9, div_10, unsqueeze_23, view_99, mul_35, view_101, addmm_10, view_103, mul_40, view_109, div_12, div_13, unsqueeze_29, view_125, mul_45, view_127, addmm_13, view_129, mul_50, view_135, div_15, div_16, unsqueeze_35, view_151, mul_55, view_153, addmm_16, view_155, mul_60, view_161, div_18, div_19, unsqueeze_41, view_177, mul_65, view_179, addmm_19, view_181, mul_70, view_187, div_21, div_22, unsqueeze_47, view_203, mul_75, view_205, addmm_22, view_207, mul_80, view_213, div_24, div_25, unsqueeze_53, view_229, mul_85, view_231, addmm_25, view_233, mul_90, view_239, div_27, div_28, unsqueeze_59, view_255, mul_95, view_257, addmm_28, view_259, cat, getitem_41, rsqrt_20, view_261, view_271, mul_103, view_273, addmm_31, view_275, mul_108, view_277, view_287, mul_111, view_289, addmm_34, view_291, mul_116, clone_167, permute_126, div_32, permute_130, permute_134, div_33, permute_138, permute_143, permute_144, alias_42, permute_145, permute_146, permute_151, div_34, permute_153, permute_157, div_35, permute_161, permute_166, permute_167, alias_43, permute_168, permute_169, permute_174, permute_176, permute_180, div_37, permute_184, permute_189, permute_190, permute_194, permute_196, permute_197, permute_206, div_41, permute_208, permute_212, div_42, permute_216, permute_221, permute_222, permute_226, permute_228, permute_229, permute_238, div_46, permute_240, permute_244, div_47, permute_248, permute_253, permute_254, permute_258, permute_260, permute_261, permute_270, div_51, permute_272, permute_276, div_52, permute_280, permute_285, permute_286, permute_290, permute_292, permute_293, permute_302, div_56, permute_304, permute_308, div_57, permute_312, permute_317, permute_318, permute_322, permute_324, permute_325, permute_334, div_61, permute_336, permute_340, div_62, permute_344, permute_349, permute_350, permute_354, permute_356, permute_357, permute_366, div_66, permute_368, permute_372, div_67, permute_376, permute_381, permute_382, permute_386, permute_388, permute_389, permute_398, div_71, permute_400, permute_404, div_72, permute_408, permute_413, permute_414, permute_418, permute_420, permute_421, permute_430, div_76, permute_432, permute_436, div_77, permute_440, permute_445, permute_446, permute_450, permute_452, permute_453, permute_462, div_81, permute_464, permute_468, div_82, permute_472, permute_477, permute_478, permute_482, permute_484, permute_485, permute_494, div_86, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('convit_base', benchmark_compiled_module)
